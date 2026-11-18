use super::{float_matmul_capability, AscendProvider};
use libloading::{Library, Symbol};
use std::{
    ffi::c_void,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, Ordering},
        Mutex,
    },
};

// CANN's aclFinalize is process-global. Managed handles cannot initialize a
// second independent runtime while one is live; share an Arc instead.
static RUNTIME_LIVE: Mutex<bool> = Mutex::new(false);

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
    quarantined: AtomicBool,
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
    RuntimeAlreadyOpen,
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
            Self::RuntimeAlreadyOpen => {
                formatter.write_str("a managed ACL runtime is already live; share its Arc")
            }
        }
    }
}

impl std::error::Error for AscendRuntimeError {}

impl AscendRuntime {
    /// Load CANN's ACL runtime, initialize it, and query the visible device count.
    pub fn open() -> Result<Self, AscendRuntimeError> {
        let path = std::env::var_os("ASCEND_ACL_LIBRARY")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("libascendcl.so"));
        Self::open_library(path)
    }

    pub fn open_library(path: impl AsRef<Path>) -> Result<Self, AscendRuntimeError> {
        let path = path.as_ref().to_path_buf();
        let mut live = RUNTIME_LIVE
            .lock()
            .map_err(|_| AscendRuntimeError::RuntimeAlreadyOpen)?;
        if *live {
            return Err(AscendRuntimeError::RuntimeAlreadyOpen);
        }
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
        let count = match query_device_count(&library) {
            Ok(count) => count,
            Err(error) => {
                finalize(&library);
                return Err(error);
            }
        };
        if count == 0 {
            finalize(&library);
            return Err(AscendRuntimeError::NoDevices);
        }
        *live = true;
        Ok(Self {
            library,
            op_library: None,
            matmul_get_workspace: None,
            matmul_execute: None,
            device_count: count,
            library_path: path,
            quarantined: AtomicBool::new(false),
        })
    }

    pub fn device_count(&self) -> u32 {
        self.device_count
    }

    pub fn library_path(&self) -> &std::path::Path {
        &self.library_path
    }

    pub(super) fn acl_library(&self) -> &Library {
        &self.library
    }
    pub(super) fn quarantined(&self) -> bool {
        self.quarantined.load(Ordering::Acquire)
    }
    pub(super) fn quarantine(&self) {
        self.quarantined.store(true, Ordering::Release);
    }
    pub(super) fn operator_library(&self) -> Option<&Library> {
        self.op_library.as_ref()
    }

    /// Load the ACLNN operator library and verify the two-phase float MatMul ABI.
    pub fn load_float_matmul_abi(&mut self) -> Result<(), AscendRuntimeError> {
        let path = std::env::var_os("ASCEND_OP_LIBRARY")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("libopapi.so"));
        self.load_float_matmul_library(path)
    }

    pub fn load_float_matmul_library(
        &mut self,
        path: impl AsRef<Path>,
    ) -> Result<(), AscendRuntimeError> {
        let path = path.as_ref();
        // SAFETY: the library is retained by `self` while symbol lookups are used.
        let library = unsafe { Library::new(path) }.map_err(|error| {
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
    /// returned plan retains a runtime borrow and is consumed by execution.
    ///
    /// # Safety
    /// Non-null pointers must be live CANN tensor handles on the same device,
    /// with mutually compatible shapes/dtypes and writable output storage.
    /// Their device allocations and tensor handles must remain valid through
    /// execution and stream synchronization. The caller must prevent conflicting
    /// accesses while CANN uses them. Null tensor handles are rejected.
    ///
    /// ```compile_fail
    /// use quantinfer_backend_ascend::AscendRuntime;
    /// fn requires_unsafe(runtime: &AscendRuntime) {
    ///     runtime.prepare_float_matmul(std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), 0);
    /// }
    /// ```
    pub unsafe fn prepare_float_matmul<'runtime>(
        &'runtime self,
        lhs: *const c_void,
        rhs: *const c_void,
        output: *mut c_void,
        cube_math_type: i8,
    ) -> Result<AscendMatmulPlan<'runtime>, AscendRuntimeError> {
        if lhs.is_null() || rhs.is_null() || output.is_null() {
            return Err(AscendRuntimeError::InvalidMatmulPlan);
        }
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
    /// This plan can be submitted only once.
    ///
    /// # Safety
    /// The workspace must be live device memory of the required size, and the
    /// stream must be valid for the plan's device. Tensor handles, tensor device
    /// allocations, workspace, runtime and its libraries must remain valid
    /// without conflicting accesses until the caller successfully synchronizes
    /// the stream after submission.
    ///
    /// ```compile_fail
    /// use quantinfer_backend_ascend::AscendMatmulPlan;
    /// fn requires_unsafe(plan: AscendMatmulPlan<'_>) {
    ///     plan.execute(std::ptr::null_mut(), std::ptr::null_mut());
    /// }
    /// ```
    pub unsafe fn execute(
        self,
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
        // SAFETY: the executor is from this runtime; the caller guarantees live
        // tensor/workspace/stream handles and their asynchronous lifetimes.
        let code = unsafe { execute(workspace, self.workspace_bytes, self.executor, stream) };
        if code != 0 {
            return Err(AscendRuntimeError::MatmulExecution(code));
        }
        Ok(())
    }
}

impl Drop for AscendRuntime {
    fn drop(&mut self) {
        let mut live = RUNTIME_LIVE
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        finalize(&self.library);
        *live = false;
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
