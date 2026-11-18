//! Linux AArch64 Q4_0 output-row path using the pinned NEON dotprod kernel.
use crate::{packing, KleidiAiBackend, KleidiAiKernel};
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{QuantizedTensor4, Tensor};
use quantinfer_kernel::{KernelBackend, KernelError, Result};
use quantinfer_runtime::{CpuScalarProvider, KernelCapability, RuntimeProvider};
use std::ffi::c_void;

unsafe extern "C" {
    fn quantinfer_kai_lhs_size(m: usize, k: usize) -> usize;
    fn quantinfer_kai_rhs_size(n: usize, k: usize) -> usize;
    fn quantinfer_kai_lhs_pack(m: usize, k: usize, lhs: *const f32, packed: *mut c_void);
    fn quantinfer_kai_rhs_pack(n: usize, k: usize, rhs: *const u8, packed: *mut c_void);
    fn quantinfer_kai_run(
        m: usize,
        n: usize,
        k: usize,
        lhs: *const c_void,
        rhs: *const c_void,
        output: *mut f32,
    );
}

#[repr(C, align(64))]
#[derive(Clone)]
struct Aligned([u8; 64]);

fn buffer(bytes: usize) -> Result<Vec<Aligned>> {
    let count = bytes
        .checked_add(63)
        .ok_or_else(|| KernelError::InvalidInput("KleidiAI allocation overflows".into()))?
        / 64;
    let mut data = Vec::new();
    data.try_reserve_exact(count)
        .map_err(|_| KernelError::InvalidInput("cannot allocate KleidiAI workspace".into()))?;
    data.resize(count, Aligned([0; 64]));
    Ok(data)
}

#[derive(Debug)]
pub struct NativeKleidiAiKernel {
    cpu: CpuScalar,
}
impl NativeKleidiAiKernel {
    pub fn open() -> Result<Self> {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return Err(KernelError::Unsupported(
                "KleidiAI requires AArch64 dotprod".into(),
            ));
        }
        Ok(Self {
            cpu: CpuScalar::new(),
        })
    }
    pub fn sdk_revision() -> &'static str {
        include_str!("../vendor-revision").trim()
    }

    fn output_rows(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        let [m, k] = lhs.shape().dims() else {
            return Err(KernelError::InvalidInput(
                "KleidiAI lhs must be a matrix".into(),
            ));
        };
        let [rhs_k, n] = rhs.shape().dims() else {
            return Err(KernelError::InvalidInput(
                "KleidiAI rhs must be a matrix".into(),
            ));
        };
        if k != rhs_k {
            return Err(KernelError::InvalidInput(
                "KleidiAI matrix dimensions disagree".into(),
            ));
        }
        let left_size = packing::packed_size(*m, *k, 34)?;
        let right_size = packing::packed_size(*n, *k, 18)?;
        let output_length = m
            .checked_mul(*n)
            .filter(|&len| len <= isize::MAX as usize / size_of::<f32>())
            .ok_or_else(|| KernelError::InvalidInput("KleidiAI output size overflows".into()))?;
        let values = lhs.as_f32()?;
        packing::validate_lhs(values)?;
        let rhs = packing::ggml_bytes(rhs)?;
        let mut left = buffer(left_size)?;
        let mut right = buffer(right_size)?;
        let mut output = Vec::new();
        output
            .try_reserve_exact(output_length)
            .map_err(|_| KernelError::InvalidInput("cannot allocate KleidiAI output".into()))?;
        output.resize(output_length, 0.0);
        // SAFETY: dimensions are nonzero, K is a multiple of 32, all products and
        // block/tile padding fit usize. These queries access no tensor pointers.
        let (sdk_left, sdk_right) = unsafe {
            (
                quantinfer_kai_lhs_size(*m, *k),
                quantinfer_kai_rhs_size(*n, *k),
            )
        };
        if (sdk_left, sdk_right) != (left_size, right_size) {
            return Err(KernelError::InvalidInput(
                "KleidiAI packed size disagrees with the pinned profile".into(),
            ));
        }
        // SAFETY: checked input lengths, finite/range-checked activations, exact
        // F16 Q4 scales/codes, 64-byte-aligned initialized workspaces of the SDK
        // sizes and a nonoverlapping initialized output slice. Construction
        // verified the required ISA. The C bridge uses the pinned header ABI.
        unsafe {
            quantinfer_kai_lhs_pack(*m, *k, values.as_ptr(), left.as_mut_ptr().cast());
            quantinfer_kai_rhs_pack(*n, *k, rhs.as_ptr(), right.as_mut_ptr().cast());
            quantinfer_kai_run(
                *m,
                *n,
                *k,
                left.as_ptr().cast(),
                right.as_ptr().cast(),
                output.as_mut_ptr(),
            );
        }
        if output.iter().any(|x| !x.is_finite()) {
            return Err(KernelError::InvalidInput(
                "KleidiAI produced nonfinite output".into(),
            ));
        }
        Ok(Tensor::from_f32(vec![*m, *n], output)?)
    }
}

impl KleidiAiKernel for NativeKleidiAiKernel {
    fn matmul_q4_0(&self, lhs: &Tensor, rhs: &QuantizedTensor4) -> Result<Tensor> {
        self.cpu.matmul_q4_0(lhs, rhs)
    }
    fn matmul_q4_0_output_rows(
        &self,
        lhs: &Tensor,
        rhs: &QuantizedTensor4,
    ) -> Option<Result<Tensor>> {
        Some(self.output_rows(lhs, rhs))
    }
    fn capabilities(&self) -> Vec<KernelCapability> {
        CpuScalarProvider::default().capabilities().to_vec()
    }
}

impl KleidiAiBackend<NativeKleidiAiKernel> {
    pub fn open() -> Result<Self> {
        NativeKleidiAiKernel::open().map(Self::new)
    }
}
