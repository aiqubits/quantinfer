#![cfg(all(feature = "native", target_os = "linux"))]
use libloading::Library;
use quantinfer_backend_ascend::{AscendBackend, AscendRuntime, CannMatmulExecutor};
use quantinfer_core::Tensor;
use quantinfer_kernel::KernelBackend;
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

struct Mock {
    library: Library,
    root: PathBuf,
}
impl Mock {
    fn compile() -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("quantinfer-acl-{}-{stamp}", std::process::id()));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("libmockacl.so");
        let output = Command::new("cc")
            .args([
                "-shared", "-fPIC", "-std=c11", "-Wall", "-Wextra", "-Werror",
            ])
            .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/mock_acl.c"))
            .arg("-o")
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        // SAFETY: freshly compiled test library with the exact local C ABI.
        let library = unsafe { Library::new(&path) }.unwrap();
        Self { library, root }
    }
    fn path(&self) -> PathBuf {
        self.root.join("libmockacl.so")
    }
    fn configure(&self, phase: i32, nth: i32) {
        // SAFETY: the exported test function has this signature; library is live.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn(i32, i32)>(b"mock_configure\0")
                .unwrap()(phase, nth);
        }
    }
    fn stat(&self, index: i32) -> i32 {
        // SAFETY: same fixed test ABI, integer input/output only.
        unsafe {
            self.library
                .get::<unsafe extern "C" fn(i32) -> i32>(b"mock_stat\0")
                .unwrap()(index)
        }
    }
    fn runtime(&self) -> Arc<AscendRuntime> {
        let mut runtime = AscendRuntime::open_library(self.path()).unwrap();
        runtime.load_float_matmul_library(self.path()).unwrap();
        Arc::new(runtime)
    }
    fn clean(&self) {
        for index in 2..=6 {
            assert_eq!(self.stat(index), 0, "resource {index}");
        }
        assert_eq!(self.stat(11), 0, "ABI/ownership violation");
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        if !self.root.as_os_str().is_empty() {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }
}

#[test]
#[ignore = "requires a configured CANN SDK, driver and Ascend device"]
fn real_cann_float_matmul_matches_the_reference_fixture() {
    let backend = AscendBackend::open(0).unwrap();
    let (lhs, rhs) = operands();
    let output = backend.matmul(&lhs, &rhs).unwrap();
    for (actual, expected) in output.as_f32().unwrap().iter().zip([5.5, 12.0, -1.5, 12.0]) {
        assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
    }
}

fn operands() -> (Tensor, Tensor) {
    (
        Tensor::from_f32(vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.5, 4.0]).unwrap(),
        Tensor::from_f32(vec![3, 2], vec![1.0, -2.0, 3.0, 4.0, -0.5, 2.0]).unwrap(),
    )
}

#[test]
fn actual_executor_obeys_the_cann_abi_and_cleans_partial_failures() {
    let mock = Mock::compile();
    // Initialization failure must finalize before another managed open succeeds.
    mock.configure(9, 0);
    assert!(AscendRuntime::open_library(mock.path()).is_err());
    assert_eq!(mock.stat(0), 1);
    assert_eq!(mock.stat(1), 1);
    mock.configure(0, 0);
    let runtime = mock.runtime();
    assert!(AscendRuntime::open_library(mock.path()).is_err());
    assert!(CannMatmulExecutor::from_runtime(runtime.clone(), -1).is_err());
    assert!(CannMatmulExecutor::from_runtime(runtime.clone(), 1).is_err());
    // SAFETY: the test-only function establishes a separate, non-owned context.
    unsafe {
        mock.library
            .get::<unsafe extern "C" fn()>(b"mock_external_context\0")
            .unwrap()();
    }
    let (lhs, rhs) = operands();
    let backend = AscendBackend::new(CannMatmulExecutor::from_runtime(runtime.clone(), 0).unwrap());
    let output = backend.matmul(&lhs, &rhs).unwrap();
    assert_eq!(output.shape().dims(), [2, 2]);
    assert_eq!(output.as_f32().unwrap(), [5.5, 12.0, -1.5, 12.0]);
    assert_eq!(mock.stat(7), 1);
    assert_eq!(mock.stat(8), 1);
    assert_eq!(mock.stat(9), 3);
    assert_eq!(mock.stat(10), 1);
    assert_eq!(mock.stat(12), 1);
    mock.clean();
    std::thread::scope(|scope| {
        let result = scope
            .spawn(|| backend.matmul(&lhs, &rhs))
            .join()
            .unwrap()
            .unwrap();
        assert_eq!(result.as_f32().unwrap(), output.as_f32().unwrap());
    });
    mock.clean();
    assert_eq!(mock.stat(12), 1);
    drop(backend);
    for (phase, nth) in [
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (4, 0),
        (6, 0),
        (7, 0),
        (8, 0),
    ] {
        mock.configure(phase, nth);
        let backend =
            AscendBackend::new(CannMatmulExecutor::from_runtime(runtime.clone(), 0).unwrap());
        assert!(
            backend.matmul(&lhs, &rhs).is_err(),
            "phase {phase}, nth {nth}"
        );
        mock.clean();
        assert_eq!(mock.stat(12), 1, "restore context after phase {phase}");
    }
    drop(runtime);
    assert_eq!(mock.stat(0), 2);
    assert_eq!(mock.stat(1), 2);
}

#[test]
fn uncertain_execution_retains_resources_and_poisoned_runtime_in_a_child() {
    if let Some(path) = std::env::var_os("QUANTINFER_ACL_CONTRACT_CHILD") {
        // SAFETY: the parent compiled this exact test library and retains it.
        let mock = Mock {
            library: unsafe { Library::new(Path::new(&path)) }.unwrap(),
            root: PathBuf::new(),
        };
        let mut runtime = AscendRuntime::open_library(Path::new(&path)).unwrap();
        runtime.load_float_matmul_library(Path::new(&path)).unwrap();
        let runtime = Arc::new(runtime);
        let phase = std::env::var("QUANTINFER_ACL_FAILURE_PHASE")
            .unwrap()
            .parse()
            .unwrap();
        if phase == 13 {
            // SAFETY: the test-only ABI establishes a live non-owned context.
            unsafe {
                mock.library
                    .get::<unsafe extern "C" fn()>(b"mock_external_context\0")
                    .unwrap()();
            }
        }
        mock.configure(phase, 0);
        let backend =
            AscendBackend::new(CannMatmulExecutor::from_runtime(runtime.clone(), 0).unwrap());
        let (lhs, rhs) = operands();
        assert!(backend.matmul(&lhs, &rhs).is_err());
        assert!(backend.matmul(&lhs, &rhs).is_err());
        assert!(CannMatmulExecutor::from_runtime(runtime.clone(), 0).is_err());
        drop((backend, runtime));
        if phase != 13 {
            assert!(mock.stat(2) > 0);
        }
        if [5, 10, 12].contains(&phase) {
            assert!(mock.stat(3) > 0);
        }
        assert_eq!(mock.stat(1), 0);
        assert_eq!(mock.stat(11), 0);
        assert!(AscendRuntime::open_library(Path::new(&path)).is_err());
        return;
    }
    let mock = Mock::compile();
    for phase in [5, 10, 11, 12, 13] {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "uncertain_execution_retains_resources_and_poisoned_runtime_in_a_child",
            ])
            .env("QUANTINFER_ACL_CONTRACT_CHILD", mock.path())
            .env("QUANTINFER_ACL_FAILURE_PHASE", phase.to_string())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
