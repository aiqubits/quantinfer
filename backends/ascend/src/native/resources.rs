//! Per-call device resources. No raw handles cross a calling-thread boundary.
use super::{
    ffi::{check, Functions, Handle},
    AscendRuntime,
};
use quantinfer_kernel::{KernelError, Result};
use std::{ffi::c_void, ptr, sync::Arc};

pub(super) struct Call {
    pub runtime: Arc<AscendRuntime>,
    api: Functions,
    context: Handle,
    previous: Handle,
    pub stream: Handle,
    tensors: Vec<Handle>,
    executors: Vec<Handle>,
    allocations: Vec<Allocation>,
    pub pending: bool,
    quarantine: bool,
}

struct Allocation {
    handle: Handle,
    bytes: usize,
}

impl Call {
    pub fn new(runtime: Arc<AscendRuntime>, api: Functions, device: i32) -> Result<Self> {
        let mut call = Self {
            runtime,
            api,
            context: ptr::null_mut(),
            previous: ptr::null_mut(),
            stream: ptr::null_mut(),
            tensors: Vec::new(),
            executors: Vec::new(),
            allocations: Vec::new(),
            pending: false,
            quarantine: false,
        };
        // SAFETY: all output pointers reference initialized local handle slots.
        let status = unsafe { (api.get_context)(&mut call.previous) };
        if status != 0 {
            call.previous = ptr::null_mut();
        }
        if status != 0 && status != 107002 {
            check("aclrtGetCurrentContext", status)?;
        }
        let mut context = ptr::null_mut();
        // SAFETY: initialized SDK and validated device; the output slot is valid.
        check("aclrtCreateContext", unsafe {
            (api.create_context)(&mut context, device)
        })?;
        call.context = context;
        if call.context.is_null() {
            return Err(KernelError::InvalidInput(
                "CANN returned a null context".into(),
            ));
        }
        // SAFETY: the handle was just created and is owned by this call.
        check("aclrtSetCurrentContext", unsafe {
            (api.set_context)(call.context)
        })?;
        let mut stream = ptr::null_mut();
        // SAFETY: the current context is live and the output slot is valid.
        check("aclrtCreateStream", unsafe {
            (api.create_stream)(&mut stream)
        })?;
        call.stream = stream;
        if call.stream.is_null() {
            return Err(KernelError::InvalidInput(
                "CANN returned a null stream".into(),
            ));
        }
        Ok(call)
    }

    pub fn allocate(&mut self, bytes: usize) -> Result<Handle> {
        if bytes == 0 {
            return Ok(ptr::null_mut());
        }
        let mut handle = ptr::null_mut();
        // SAFETY: the current context is live; 0 is ACL_MEM_MALLOC_HUGE_FIRST.
        check("aclrtMalloc", unsafe {
            (self.api.allocate)(&mut handle, bytes, 0)
        })?;
        if handle.is_null() {
            return Err(KernelError::InvalidInput(
                "CANN returned a null device allocation".into(),
            ));
        }
        self.allocations.push(Allocation { handle, bytes });
        Ok(handle)
    }

    pub fn tensor(&mut self, shape: [usize; 2], host: Option<&[f32]>) -> Result<Handle> {
        let count = shape[0]
            .checked_mul(shape[1])
            .ok_or_else(|| KernelError::InvalidInput("Ascend tensor length overflows".into()))?;
        let bytes = count
            .checked_mul(size_of::<f32>())
            .ok_or_else(|| KernelError::InvalidInput("Ascend tensor byte size overflows".into()))?;
        let shape = shape.map(|value| {
            i64::try_from(value)
                .map_err(|_| KernelError::InvalidInput("Ascend dimension exceeds i64".into()))
        });
        let [rows, columns] = shape;
        let shape = [rows?, columns?];
        let allocation = self.allocate(bytes)?;
        if let Some(host) = host {
            if host.len() != count {
                return Err(KernelError::InvalidInput(
                    "Ascend host tensor length mismatch".into(),
                ));
            }
            // SAFETY: both ranges have exactly `bytes` bytes. 1 is synchronous H2D.
            check("aclrtMemcpy H2D", unsafe {
                (self.api.copy)(allocation, bytes, host.as_ptr().cast::<c_void>(), bytes, 1)
            })?;
        }
        let strides = [shape[1], 1];
        // SAFETY: valid F32 (0), ND (2), dense shape/strides, and owned device
        // allocation. CANN copies the shape/stride descriptors on creation.
        let tensor = unsafe {
            (self.api.tensor)(
                shape.as_ptr(),
                2,
                0,
                strides.as_ptr(),
                0,
                2,
                shape.as_ptr(),
                2,
                allocation,
            )
        };
        if tensor.is_null() {
            return Err(KernelError::InvalidInput(
                "aclCreateTensor returned null".into(),
            ));
        }
        self.tensors.push(tensor);
        Ok(tensor)
    }

    pub fn synchronize(&mut self) -> Result<()> {
        // SAFETY: this call owns the live stream and all queued operand storage.
        check("aclrtSynchronizeStream", unsafe {
            (self.api.sync)(self.stream)
        })?;
        self.pending = false;
        Ok(())
    }

    pub fn matmul(&mut self, a: Handle, b: Handle, c: Handle) -> Result<()> {
        let mut workspace_bytes = 0u64;
        let mut executor = ptr::null_mut();
        // SAFETY: owned compatible dense F32 tensors, initialized output slots;
        // 0 requests KEEP_DTYPE and never silently reduces input precision.
        check("aclnnMmGetWorkspaceSize", unsafe {
            (self.api.prepare)(a, b, c, 0, &mut workspace_bytes, &mut executor)
        })?;
        if executor.is_null() {
            return Err(KernelError::InvalidInput(
                "CANN returned a null MatMul executor".into(),
            ));
        }
        // SAFETY: immediately following the successful first phase, as required
        // by CANN. Explicit ownership allows cancellation if workspace allocation
        // fails, and destruction only after asynchronous execution is complete.
        let status = unsafe { (self.api.repeatable)(executor) };
        if status != 0 {
            self.quarantine = true;
            return check("aclSetAclOpExecutorRepeatable", status);
        }
        self.executors.push(executor);
        let length = usize::try_from(workspace_bytes)
            .map_err(|_| KernelError::InvalidInput("Ascend workspace exceeds usize".into()))?;
        let workspace = self.allocate(length)?;
        self.pending = true;
        // SAFETY: all operands, executor, workspace and stream are owned by this
        // call and survive through stream synchronization, including failures.
        check("aclnnMm", unsafe {
            (self.api.execute)(workspace, workspace_bytes, executor, self.stream)
        })
    }

    pub fn output(&mut self, tensor: usize, values: &mut [f32]) -> Result<()> {
        // Device allocations for the three tensors precede the workspace.
        let bytes = std::mem::size_of_val(values);
        let source = self.allocations.get(tensor).ok_or_else(|| {
            KernelError::InvalidInput("Ascend output allocation is missing".into())
        })?;
        if self.pending || bytes > source.bytes {
            return Err(KernelError::InvalidInput(
                "Ascend output copy exceeds synchronized allocation".into(),
            ));
        }
        // SAFETY: the destination is a writable slice, source allocation has the
        // requested length, stream synchronization succeeded. 2 is synchronous D2H.
        check("aclrtMemcpy D2H", unsafe {
            (self.api.copy)(values.as_mut_ptr().cast(), bytes, source.handle, bytes, 2)
        })
    }

    fn release_status(&mut self, api: &str, status: i32) -> Result<()> {
        if status != 0 {
            self.quarantine = true;
        }
        check(api, status)
    }

    pub fn close(&mut self) -> Result<()> {
        if self.quarantine {
            return Err(KernelError::InvalidInput(
                "Ascend resource ownership is uncertain".into(),
            ));
        }
        if self.pending {
            self.synchronize()?;
        }
        // SAFETY: no queued work remains. Release in dependency order; retain
        // all remaining resources on the first failed destructor, since retrying
        // a handle after an SDK error could double-free an already released object.
        while let Some(&executor) = self.executors.last() {
            let status = unsafe { (self.api.destroy_executor)(executor) };
            self.release_status("aclDestroyAclOpExecutor", status)?;
            self.executors.pop();
        }
        while let Some(&tensor) = self.tensors.last() {
            let status = unsafe { (self.api.destroy_tensor)(tensor) };
            self.release_status("aclDestroyTensor", status)?;
            self.tensors.pop();
        }
        while let Some(allocation) = self.allocations.last() {
            let status = unsafe { (self.api.free)(allocation.handle) };
            self.release_status("aclrtFree", status)?;
            self.allocations.pop();
        }
        if !self.stream.is_null() {
            let status = unsafe { (self.api.destroy_stream)(self.stream) };
            self.release_status("aclrtDestroyStream", status)?;
            self.stream = ptr::null_mut();
        }
        if !self.context.is_null() {
            let status = unsafe { (self.api.destroy_context)(self.context) };
            self.release_status("aclrtDestroyContext", status)?;
            self.context = ptr::null_mut();
        }
        if !self.previous.is_null() {
            let status = unsafe { (self.api.set_context)(self.previous) };
            self.release_status("aclrtSetCurrentContext restore", status)?;
            self.previous = ptr::null_mut();
        }
        Ok(())
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        if self.quarantine || self.close().is_err() {
            // Unknown completion state: retain device resources and library
            // lifetime rather than freeing storage still used by an NPU task.
            self.runtime.quarantine();
            std::mem::forget(self.runtime.clone());
        }
    }
}
