//! CANN runtime and ACL tensor signatures, isolated from resource ownership.
use super::AscendRuntime;
use libloading::Library;
use quantinfer_kernel::{KernelError, Result};
use std::ffi::c_void;

pub(super) type Handle = *mut c_void;
type GetContext = unsafe extern "C" fn(*mut Handle) -> i32;
type CreateContext = unsafe extern "C" fn(*mut Handle, i32) -> i32;
type SetContext = unsafe extern "C" fn(Handle) -> i32;
type CreateStream = unsafe extern "C" fn(*mut Handle) -> i32;
type ResourceOp = unsafe extern "C" fn(Handle) -> i32;
type Allocate = unsafe extern "C" fn(*mut Handle, usize, i32) -> i32;
type Memcpy = unsafe extern "C" fn(Handle, usize, *const c_void, usize, i32) -> i32;
type CreateTensor = unsafe extern "C" fn(
    *const i64,
    u64,
    i32,
    *const i64,
    i64,
    i32,
    *const i64,
    u64,
    Handle,
) -> Handle;
type Prepare =
    unsafe extern "C" fn(*const c_void, *const c_void, Handle, i8, *mut u64, *mut Handle) -> i32;
type Execute = unsafe extern "C" fn(Handle, u64, Handle, Handle) -> i32;

#[derive(Clone, Copy)]
pub(super) struct Functions {
    pub get_context: GetContext,
    pub create_context: CreateContext,
    pub set_context: SetContext,
    pub destroy_context: ResourceOp,
    pub create_stream: CreateStream,
    pub sync: ResourceOp,
    pub destroy_stream: ResourceOp,
    pub allocate: Allocate,
    pub free: ResourceOp,
    pub copy: Memcpy,
    pub tensor: CreateTensor,
    pub destroy_tensor: ResourceOp,
    pub prepare: Prepare,
    pub execute: Execute,
    pub repeatable: ResourceOp,
    pub destroy_executor: ResourceOp,
}

fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
    // SAFETY: every call site specifies the published CANN C signature. An Arc
    // to the owning runtime keeps both libraries loaded during all uses.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            KernelError::Unsupported(format!(
                "missing CANN API {}: {error}",
                String::from_utf8_lossy(name)
            ))
        })
}

impl Functions {
    pub fn load(runtime: &AscendRuntime) -> Result<Self> {
        let acl = runtime.acl_library();
        let op = runtime
            .operator_library()
            .ok_or_else(|| KernelError::Unsupported("ACLNN MatMul ABI is not loaded".into()))?;
        Ok(Self {
            get_context: symbol(acl, b"aclrtGetCurrentContext\0")?,
            create_context: symbol(acl, b"aclrtCreateContext\0")?,
            set_context: symbol(acl, b"aclrtSetCurrentContext\0")?,
            destroy_context: symbol(acl, b"aclrtDestroyContext\0")?,
            create_stream: symbol(acl, b"aclrtCreateStream\0")?,
            sync: symbol(acl, b"aclrtSynchronizeStream\0")?,
            destroy_stream: symbol(acl, b"aclrtDestroyStream\0")?,
            allocate: symbol(acl, b"aclrtMalloc\0")?,
            free: symbol(acl, b"aclrtFree\0")?,
            copy: symbol(acl, b"aclrtMemcpy\0")?,
            tensor: symbol(op, b"aclCreateTensor\0")?,
            destroy_tensor: symbol(op, b"aclDestroyTensor\0")?,
            prepare: symbol(op, b"aclnnMmGetWorkspaceSize\0")?,
            execute: symbol(op, b"aclnnMm\0")?,
            repeatable: symbol(op, b"aclSetAclOpExecutorRepeatable\0")?,
            destroy_executor: symbol(op, b"aclDestroyAclOpExecutor\0")?,
        })
    }
}

pub(super) fn check(api: &str, code: i32) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(KernelError::InvalidInput(format!(
            "{api} failed with CANN code {code}"
        )))
    }
}
