use std::collections::BTreeSet;
use std::ffi::c_void;
use std::path::PathBuf;

use libloading::{Library, Symbol};
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{
    DType, Device, QuantizationScheme, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K,
    QuantizedTensorQ5K, QuantizedTensorQ6K, Tensor,
};
use quantinfer_kernel::{KernelBackend, Result as KernelResult};
use quantinfer_runtime::{KernelCapability, Operation, RuntimeProvider};

/// Runtime planning capabilities exposed after ACLNN MatMul symbols are found.
#[derive(Debug, Clone)]
pub struct AscendProvider {
    capabilities: Vec<KernelCapability>,
}

impl RuntimeProvider for AscendProvider {
    fn name(&self) -> &str {
        "ascend"
    }

    fn capabilities(&self) -> &[KernelCapability] {
        &self.capabilities
    }
}

/// The execution hook supplied by an ACL Tensor adapter. The adapter owns
/// tensor creation, device memory and stream synchronization; this crate keeps
/// the model/runtime boundary independent of those CANN details.
pub trait AscendMatmulExecutor: std::fmt::Debug + Send + Sync {
    fn matmul_f32(&self, lhs: &Tensor, rhs: &Tensor) -> KernelResult<Tensor>;
}

/// Runtime backend for an Ascend executor with explicit CPU fallbacks for
/// operators not yet represented by the ACLNN adapter.
#[derive(Debug)]
pub struct AscendBackend<E> {
    executor: E,
    fallback: CpuScalar,
    capabilities: Vec<KernelCapability>,
}

impl<E: AscendMatmulExecutor> AscendBackend<E> {
    pub fn new(executor: E) -> Self {
        let provider = AscendProvider {
            capabilities: vec![float_matmul_capability()],
        };
        Self {
            executor,
            fallback: CpuScalar::new(),
            capabilities: provider.capabilities.clone(),
        }
    }
}

impl<E: AscendMatmulExecutor> RuntimeProvider for AscendBackend<E> {
    fn name(&self) -> &str {
        "ascend"
    }

    fn capabilities(&self) -> &[KernelCapability] {
        &self.capabilities
    }
}

macro_rules! fallback_kernel {
    ($($method:ident($($arg:ident : $ty:ty),*) -> $ret:ty),* $(,)?) => {
        $(fn $method(&self, $($arg: $ty),*) -> $ret {
            self.fallback.$method($($arg),*)
        })*
    };
}

impl<E: AscendMatmulExecutor> KernelBackend for AscendBackend<E> {
    fn matmul(&self, lhs: &Tensor, rhs: &Tensor) -> KernelResult<Tensor> {
        self.executor.matmul_f32(lhs, rhs)
    }

    fallback_kernel! {
        reuse_workspace() -> bool,
        matmul_q8_0(lhs: &Tensor, rhs: &QuantizedTensor) -> KernelResult<Tensor>,
        matmul_q4_0(lhs: &Tensor, rhs: &QuantizedTensor4) -> KernelResult<Tensor>,
        matmul_q8_0_output_rows(lhs: &Tensor, rhs: &QuantizedTensor) -> KernelResult<Tensor>,
        matmul_q4_0_output_rows(lhs: &Tensor, rhs: &QuantizedTensor4) -> KernelResult<Tensor>,
        matmul_q4_k(lhs: &Tensor, rhs: &QuantizedTensorQ4K) -> KernelResult<Tensor>,
        matmul_q5_k(lhs: &Tensor, rhs: &QuantizedTensorQ5K) -> KernelResult<Tensor>,
        matmul_q6_k(lhs: &Tensor, rhs: &QuantizedTensorQ6K) -> KernelResult<Tensor>,
        rms_norm(input: &Tensor, weight: &Tensor, epsilon: f32) -> KernelResult<Tensor>,
        rms_norm_heads(input: &Tensor, weight: &Tensor, epsilon: f32) -> KernelResult<Tensor>,
        softmax(input: &Tensor) -> KernelResult<Tensor>,
        silu_mul(gate: &Tensor, up: &Tensor) -> KernelResult<Tensor>,
        apply_rope(input: &Tensor, cos: &[f32], sin: &[f32]) -> KernelResult<Tensor>,
        embedding(table: &Tensor, token_ids: &[usize]) -> KernelResult<Tensor>,
    }
}

/// CANN runtime handle used to verify that an Ascend device is available and to
/// invoke the two-phase ACLNN MatMul ABI. ACL Tensor creation, device-memory
/// ownership, and model-level NPU scheduling remain the responsibility of the
/// executor supplied to `AscendBackend`.
pub struct AscendRuntime {
    library: Library,
    op_library: Option<Library>,
    matmul_get_workspace: Option<
        unsafe extern "C" fn(
            *const c_void,
            *const c_void,
            *mut c_void,
            i8,
            *mut u64,
            *mut *mut c_void,
        ) -> i32,
    >,
    matmul_execute: Option<unsafe extern "C" fn(*mut c_void, u64, *mut c_void, *mut c_void) -> i32>,
    device_count: u32,
    library_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AscendRuntimeError {
    LibraryUnavailable(String),
    Initialization(i32),
    DeviceQuery(i32),
    NoDevices,
    OperatorLibraryUnavailable(String),
    MatmulAbiUnavailable(String),
    MatmulWorkspace(i32),
    MatmulExecution(i32),
    InvalidMatmulPlan,
}

impl std::fmt::Display for AscendRuntimeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LibraryUnavailable(message) => formatter.write_str(message),
            Self::Initialization(code) => write!(formatter, "aclInit failed with code {code}"),
            Self::DeviceQuery(code) => {
                write!(formatter, "aclrtGetDeviceCount failed with code {code}")
            }
            Self::NoDevices => formatter.write_str("CANN reports no Ascend devices"),
            Self::OperatorLibraryUnavailable(message) => formatter.write_str(message),
            Self::MatmulAbiUnavailable(message) => formatter.write_str(message),
            Self::MatmulWorkspace(code) => {
                write!(formatter, "aclnnMmGetWorkspaceSize failed with code {code}")
            }
            Self::MatmulExecution(code) => write!(formatter, "aclnnMm failed with code {code}"),
            Self::InvalidMatmulPlan => formatter.write_str("ACLNN returned an invalid MatMul plan"),
        }
    }
}

impl std::error::Error for AscendRuntimeError {}

impl AscendRuntime {
    /// Load CANN's ACL runtime, initialize it, and query the visible device count.
    pub fn open() -> Result<Self, AscendRuntimeError> {
        let path = std::env::var_os("ASCEND_ACL_LIBRARY")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("libascendcl.so"));
        // SAFETY: the library remains owned by this handle for the lifetime of all symbols.
        let library = unsafe { Library::new(&path) }.map_err(|error| {
            AscendRuntimeError::LibraryUnavailable(format!(
                "cannot load CANN ACL library {}: {error}",
                path.display()
            ))
        })?;
        // SAFETY: CANN exports these symbols with the documented C ABI.
        let init: Symbol<'_, unsafe extern "C" fn(*const std::ffi::c_char) -> i32> =
            unsafe { library.get(b"aclInit\0") }
                .map_err(|error| AscendRuntimeError::LibraryUnavailable(error.to_string()))?;
        // SAFETY: a null configuration path requests the default CANN configuration.
        let code = unsafe { init(std::ptr::null()) };
        if code != 0 {
            return Err(AscendRuntimeError::Initialization(code));
        }
        let count = query_device_count(&library)?;
        if count == 0 {
            finalize(&library);
            return Err(AscendRuntimeError::NoDevices);
        }
        Ok(Self {
            library,
            op_library: None,
            matmul_get_workspace: None,
            matmul_execute: None,
            device_count: count,
            library_path: path,
        })
    }

    pub fn device_count(&self) -> u32 {
        self.device_count
    }

    pub fn library_path(&self) -> &std::path::Path {
        &self.library_path
    }

    /// Load the ACLNN operator library and verify the two-phase float MatMul ABI.
    pub fn load_float_matmul_abi(&mut self) -> Result<(), AscendRuntimeError> {
        let path = std::env::var_os("ASCEND_OP_LIBRARY")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("libopapi.so"));
        // SAFETY: the library is retained by `self` while symbol lookups are used.
        let library = unsafe { Library::new(&path) }.map_err(|error| {
            AscendRuntimeError::OperatorLibraryUnavailable(format!(
                "cannot load CANN operator library {}: {error}",
                path.display()
            ))
        })?;
        type GetWorkspace = unsafe extern "C" fn(
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *mut std::ffi::c_void,
            i8,
            *mut u64,
            *mut *mut std::ffi::c_void,
        ) -> i32;
        // SAFETY: the symbol type matches the documented ACLNN C ABI.
        let get_workspace: GetWorkspace = *unsafe { library.get(b"aclnnMmGetWorkspaceSize\0") }
            .map_err(|error| AscendRuntimeError::MatmulAbiUnavailable(error.to_string()))?;
        type Execute = unsafe extern "C" fn(
            *mut std::ffi::c_void,
            u64,
            *mut std::ffi::c_void,
            *mut std::ffi::c_void,
        ) -> i32;
        // SAFETY: the symbol type matches the documented ACLNN second-phase ABI.
        let execute: Execute = *unsafe { library.get(b"aclnnMm\0") }
            .map_err(|error| AscendRuntimeError::MatmulAbiUnavailable(error.to_string()))?;
        self.op_library = Some(library);
        self.matmul_get_workspace = Some(get_workspace);
        self.matmul_execute = Some(execute);
        Ok(())
    }

    pub fn has_float_matmul_abi(&self) -> bool {
        self.op_library.is_some()
            && self.matmul_get_workspace.is_some()
            && self.matmul_execute.is_some()
    }

    /// Prepare an ACLNN MatMul by running its first (workspace-size) phase.
    ///
    /// The tensor pointers are opaque handles created by CANN's ACL tensor API. The
    /// returned plan owns the executor handle and must be executed before the
    /// runtime is dropped.
    pub fn prepare_float_matmul<'runtime>(
        &'runtime self,
        lhs: *const c_void,
        rhs: *const c_void,
        output: *mut c_void,
        cube_math_type: i8,
    ) -> Result<AscendMatmulPlan<'runtime>, AscendRuntimeError> {
        let get_workspace = self
            .matmul_get_workspace
            .ok_or(AscendRuntimeError::InvalidMatmulPlan)?;
        let mut workspace_bytes = 0_u64;
        let mut executor = std::ptr::null_mut();
        // SAFETY: callers provide live ACL tensor handles and writable output
        // storage for the workspace size and executor returned by CANN.
        let code = unsafe {
            get_workspace(
                lhs,
                rhs,
                output,
                cube_math_type,
                &mut workspace_bytes,
                &mut executor,
            )
        };
        if code != 0 {
            return Err(AscendRuntimeError::MatmulWorkspace(code));
        }
        if executor.is_null() {
            return Err(AscendRuntimeError::InvalidMatmulPlan);
        }
        Ok(AscendMatmulPlan {
            runtime: self,
            workspace_bytes,
            executor,
        })
    }

    /// Build the NPU execution plan provider after the ACLNN ABI has been verified.
    pub fn execution_provider(&self) -> Option<AscendProvider> {
        self.op_library.as_ref()?;
        Some(AscendProvider {
            capabilities: vec![float_matmul_capability()],
        })
    }
}

fn float_matmul_capability() -> KernelCapability {
    KernelCapability {
        device: Device::Npu,
        dtype: DType::F32,
        quantization: QuantizationScheme::None,
        tensor_layout: "row_major".to_owned(),
        supported_operations: BTreeSet::from([Operation::Matmul]),
        alignment: 1,
        head_dimension: None,
        maximum_sequence_length: None,
    }
}

/// An ACLNN MatMul executor returned by `aclnnMmGetWorkspaceSize`.
pub struct AscendMatmulPlan<'runtime> {
    runtime: &'runtime AscendRuntime,
    workspace_bytes: u64,
    executor: *mut c_void,
}

impl AscendMatmulPlan<'_> {
    pub fn workspace_bytes(&self) -> u64 {
        self.workspace_bytes
    }

    /// Execute the prepared MatMul on an ACL runtime stream.
    ///
    /// `workspace` must point to at least `workspace_bytes()` bytes allocated by
    /// the caller. A null pointer is accepted when CANN reports a zero-sized
    /// workspace. `stream` is the ACL stream handle for the target device.
    pub fn execute(
        &self,
        workspace: *mut c_void,
        stream: *mut c_void,
    ) -> Result<(), AscendRuntimeError> {
        if self.workspace_bytes != 0 && workspace.is_null() {
            return Err(AscendRuntimeError::InvalidMatmulPlan);
        }
        let execute = self
            .runtime
            .matmul_execute
            .ok_or(AscendRuntimeError::InvalidMatmulPlan)?;
        // SAFETY: the executor was created by CANN for this runtime and the
        // workspace/stream handles are validated by the CANN operator.
        let code = unsafe { execute(workspace, self.workspace_bytes, self.executor, stream) };
        if code != 0 {
            return Err(AscendRuntimeError::MatmulExecution(code));
        }
        Ok(())
    }
}

impl Drop for AscendRuntime {
    fn drop(&mut self) {
        finalize(&self.library);
    }
}

fn query_device_count(library: &Library) -> Result<u32, AscendRuntimeError> {
    let mut count = 0_u32;
    // SAFETY: CANN writes one u32 to the valid local pointer.
    let query: Symbol<'_, unsafe extern "C" fn(*mut u32) -> i32> =
        unsafe { library.get(b"aclrtGetDeviceCount\0") }
            .map_err(|error| AscendRuntimeError::LibraryUnavailable(error.to_string()))?;
    // SAFETY: the pointer points to initialized writable storage.
    let code = unsafe { query(&mut count) };
    if code != 0 {
        return Err(AscendRuntimeError::DeviceQuery(code));
    }
    Ok(count)
}

fn finalize(library: &Library) {
    // SAFETY: finalization is called only after successful aclInit on this handle.
    let Ok(finalize): Result<Symbol<'_, unsafe extern "C" fn() -> i32>, _> =
        (unsafe { library.get(b"aclFinalize\0") })
    else {
        return;
    };
    // SAFETY: the ACL runtime was initialized by this handle.
    let _ = unsafe { finalize() };
}

#[cfg(test)]
mod tests {
    use quantinfer_core::Tensor;
    use quantinfer_kernel::KernelBackend;
    use quantinfer_runtime::RuntimeProvider;

    use super::{
        AscendBackend, AscendMatmulExecutor, AscendRuntime, AscendRuntimeError, CpuScalar,
    };

    #[test]
    fn missing_library_is_reported_without_panicking() {
        std::env::set_var("ASCEND_ACL_LIBRARY", "/missing/libascendcl.so");
        let result = AscendRuntime::open();
        assert!(matches!(
            result,
            Err(AscendRuntimeError::LibraryUnavailable(_))
        ));
    }

    #[test]
    fn operator_library_is_not_assumed_when_acl_is_missing() {
        std::env::set_var("ASCEND_ACL_LIBRARY", "/missing/libascendcl.so");
        std::env::set_var("ASCEND_OP_LIBRARY", "/missing/libopapi.so");
        assert!(AscendRuntime::open().is_err());
    }

    #[derive(Debug)]
    struct ReferenceExecutor;

    impl AscendMatmulExecutor for ReferenceExecutor {
        fn matmul_f32(&self, lhs: &Tensor, rhs: &Tensor) -> quantinfer_kernel::Result<Tensor> {
            CpuScalar::new().matmul(lhs, rhs)
        }
    }

    #[test]
    fn runtime_backend_routes_float_matmul_to_executor() {
        let backend = AscendBackend::new(ReferenceExecutor);
        let lhs = Tensor::from_f32(vec![1, 2], vec![2.0, 3.0]).unwrap();
        let rhs = Tensor::from_f32(vec![2, 1], vec![4.0, 5.0]).unwrap();
        let output = backend.matmul(&lhs, &rhs).unwrap();
        assert_eq!(output.as_f32().unwrap(), &[23.0]);
        assert_eq!(backend.name(), "ascend");
    }
}
