//! A native LUT4 profile following the T-MAC bit-plane/lookup method. It does
//! not invoke the upstream TVM compiler or claim its tuning/performance results.
mod lookup;
mod weights;
use crate::{TmacBackend, TmacKernel};
pub use lookup::LutIsa;
use lookup::Selector;
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{QuantizedTensor4, Tensor};
use quantinfer_kernel::{KernelBackend, KernelError, Result};
use quantinfer_runtime::{CpuScalarProvider, KernelCapability, RuntimeProvider};
pub use weights::Lut4Weights;

pub struct NativeLut4Kernel {
    selector: Selector,
    cpu: CpuScalar,
}
impl std::fmt::Debug for NativeLut4Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeLut4Kernel")
            .field("isa", &self.isa())
            .finish()
    }
}
impl Default for NativeLut4Kernel {
    fn default() -> Self {
        Self::new()
    }
}
impl NativeLut4Kernel {
    pub fn new() -> Self {
        Self {
            selector: Selector::detect(),
            cpu: CpuScalar::new(),
        }
    }
    pub fn scalar() -> Self {
        Self {
            selector: Selector::scalar(),
            cpu: CpuScalar::new(),
        }
    }
    pub fn isa(&self) -> LutIsa {
        self.selector.isa
    }

    pub fn matmul_prepared(&self, lhs: &Tensor, rhs: &Lut4Weights) -> Result<Tensor> {
        let [m, k] = lhs.shape().dims() else {
            return Err(KernelError::InvalidInput(
                "LUT4 lhs must be a matrix".into(),
            ));
        };
        if *m == 0 || *k != rhs.k {
            return Err(KernelError::InvalidInput(
                "LUT4 matrix dimensions disagree".into(),
            ));
        }
        let values = lhs.as_f32()?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(KernelError::InvalidInput(
                "LUT4 activations must be finite".into(),
            ));
        }
        let count = m
            .checked_mul(rhs.n)
            .ok_or_else(|| KernelError::InvalidInput("LUT4 output size overflows".into()))?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(count)
            .map_err(|_| KernelError::InvalidInput("cannot allocate LUT4 output".into()))?;
        output.resize(count, 0.0);
        let mut tables = Vec::new();
        tables
            .try_reserve_exact(k / 4)
            .map_err(|_| KernelError::InvalidInput("cannot allocate LUT4 tables".into()))?;
        tables.resize(k / 4, [0.0f32; 16]);
        for row in 0..*m {
            for (table, x) in tables
                .iter_mut()
                .zip(values[row * k..(row + 1) * k].as_chunks::<4>().0)
            {
                table[0] = 0.0;
                for pattern in 1usize..16 {
                    let bit = pattern.trailing_zeros() as usize;
                    table[pattern] = table[pattern & (pattern - 1)] + x[bit];
                }
                if table.iter().any(|value| !value.is_finite()) {
                    return Err(KernelError::InvalidInput(
                        "LUT4 activation table overflow".into(),
                    ));
                }
            }
            for column in 0..rhs.n {
                let mut sum = 0.0;
                for block in 0..k / 32 {
                    let mut unscaled = 0.0;
                    for group in 0..8 {
                        let index = block * 8 + group;
                        unscaled += self
                            .selector
                            .dot(&tables[index], rhs.planes[column * (k / 4) + index]);
                    }
                    sum += unscaled * rhs.scales[column * (k / 32) + block];
                }
                if !sum.is_finite() {
                    return Err(KernelError::InvalidInput("LUT4 matmul overflow".into()));
                }
                output[row * rhs.n + column] = sum;
            }
        }
        Ok(Tensor::from_f32(vec![*m, rhs.n], output)?)
    }
}

impl TmacKernel for NativeLut4Kernel {
    fn matmul_q4_0(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        self.cpu.matmul_q4_0(lhs, rhs)
    }
    fn matmul_q4_0_output_rows(
        &self,
        lhs: &Tensor,
        rhs: &QuantizedTensor4,
    ) -> Option<Result<Tensor>> {
        Some(
            Lut4Weights::from_output_rows(rhs)
                .and_then(|packed| self.matmul_prepared(lhs, &packed)),
        )
    }
    fn capabilities(&self) -> Vec<KernelCapability> {
        CpuScalarProvider::default().capabilities().to_vec()
    }
}

impl TmacBackend<NativeLut4Kernel> {
    pub fn open() -> Result<Self> {
        Ok(Self::new(NativeLut4Kernel::new()))
    }
    pub fn implementation(&self) -> LutIsa {
        self.kernel.isa()
    }
}
