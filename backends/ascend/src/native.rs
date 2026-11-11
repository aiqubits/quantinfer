use std::path::PathBuf;

use libloading::{Library, Symbol};

/// CANN runtime handle used to verify that an Ascend device is available.
/// Operator execution is added after the ACL tensor and operator ABI is wired.
pub struct AscendRuntime {
    library: Library,
    op_library: Option<Library>,
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
        // SAFETY: the symbol type matches the documented ACLNN C ABI and is only probed here.
        unsafe {
            library.get::<unsafe extern "C" fn(
                *const std::ffi::c_void,
                *const std::ffi::c_void,
                *mut std::ffi::c_void,
                i8,
                *mut u64,
                *mut *mut std::ffi::c_void,
            ) -> i32>(b"aclnnMmGetWorkspaceSize\0")
        }
        .map_err(|error| AscendRuntimeError::MatmulAbiUnavailable(error.to_string()))?;
        // SAFETY: the symbol type matches the documented ACLNN second-phase ABI and is only probed.
        unsafe {
            library.get::<unsafe extern "C" fn(
                *mut std::ffi::c_void,
                u64,
                *mut std::ffi::c_void,
                *mut std::ffi::c_void,
            ) -> i32>(b"aclnnMm\0")
        }
        .map_err(|error| AscendRuntimeError::MatmulAbiUnavailable(error.to_string()))?;
        self.op_library = Some(library);
        Ok(())
    }

    pub fn has_float_matmul_abi(&self) -> bool {
        self.op_library.is_some()
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
    use super::{AscendRuntime, AscendRuntimeError};

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
}
