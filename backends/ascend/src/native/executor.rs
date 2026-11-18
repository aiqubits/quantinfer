//! Concrete row-major F32 ACLNN MatMul executor.
use super::{ffi::Functions, resources::Call, AscendMatmulExecutor, AscendRuntime};
use quantinfer_core::Tensor;
use quantinfer_kernel::{KernelError, Result};
use std::{
    fmt,
    sync::{Arc, Mutex},
};

pub struct CannMatmulExecutor {
    runtime: Arc<AscendRuntime>,
    api: Functions,
    device: i32,
    healthy: Mutex<bool>,
}

impl fmt::Debug for CannMatmulExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CannMatmulExecutor")
            .field("device", &self.device)
            .finish()
    }
}

impl CannMatmulExecutor {
    pub fn open(device: i32) -> Result<Self> {
        let mut runtime =
            AscendRuntime::open().map_err(|e| KernelError::Unsupported(e.to_string()))?;
        runtime
            .load_float_matmul_abi()
            .map_err(|e| KernelError::Unsupported(e.to_string()))?;
        Self::from_runtime(Arc::new(runtime), device)
    }

    /// Share the initialized runtime; the ACLNN operator ABI must be loaded.
    /// F32 KEEP_DTYPE is requested. Unsupported devices return the SDK error;
    /// input precision is never silently reduced.
    pub fn from_runtime(runtime: Arc<AscendRuntime>, device: i32) -> Result<Self> {
        if runtime.quarantined() {
            return Err(KernelError::InvalidInput(
                "Ascend runtime is quarantined after an uncertain execution; restart required"
                    .into(),
            ));
        }
        if device < 0 || device as u32 >= runtime.device_count() {
            return Err(KernelError::InvalidInput(
                "Ascend device index is out of range".into(),
            ));
        }
        let api = Functions::load(&runtime)?;
        Ok(Self {
            runtime,
            api,
            device,
            healthy: Mutex::new(true),
        })
    }
}

impl AscendMatmulExecutor for CannMatmulExecutor {
    fn matmul_f32(&self, lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        let [m, k] = lhs.shape().dims() else {
            return Err(KernelError::InvalidInput(
                "Ascend lhs must be a matrix".into(),
            ));
        };
        let [rhs_k, n] = rhs.shape().dims() else {
            return Err(KernelError::InvalidInput(
                "Ascend rhs must be a matrix".into(),
            ));
        };
        if k != rhs_k || *m == 0 || *k == 0 || *n == 0 {
            return Err(KernelError::InvalidInput(
                "Ascend matmul requires nonempty compatible dimensions".into(),
            ));
        }
        let lhs_values = lhs.as_f32()?;
        let rhs_values = rhs.as_f32()?;
        let mut healthy = self
            .healthy
            .lock()
            .map_err(|_| KernelError::InvalidInput("Ascend executor lock poisoned".into()))?;
        if !*healthy || self.runtime.quarantined() {
            return Err(KernelError::InvalidInput(
                if self.runtime.quarantined() {
                    "Ascend runtime is quarantined; restart required"
                } else {
                    "Ascend executor requires reopening after an execution failure"
                }
                .into(),
            ));
        }
        let length = m
            .checked_mul(*n)
            .ok_or_else(|| KernelError::InvalidInput("Ascend output length overflows".into()))?;
        let mut output = Vec::new();
        output.try_reserve_exact(length).map_err(|e| {
            KernelError::InvalidInput(format!("cannot allocate Ascend host output: {e}"))
        })?;
        output.resize(length, 0.0);
        let mut call = Call::new(self.runtime.clone(), self.api, self.device)?;
        let a = call.tensor([*m, *k], Some(lhs_values))?;
        let b = call.tensor([*k, *n], Some(rhs_values))?;
        let c = call.tensor([*m, *n], None)?;
        if let Err(error) = call.matmul(a, b, c) {
            *healthy = false;
            return Err(error);
        }
        if let Err(error) = call.synchronize() {
            *healthy = false;
            return Err(error);
        }
        call.output(2, &mut output)?;
        call.close()?;
        Ok(Tensor::from_f32(vec![*m, *n], output)?)
    }
}
