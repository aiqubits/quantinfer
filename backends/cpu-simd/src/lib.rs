//! Runtime-checked SIMD kernels for the CPU backend.

#![forbid(unsafe_op_in_unsafe_fn)]

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdIsa {
    Avx2,
    Avx512,
    Neon,
}

type ScaledAddFn = unsafe fn(&mut [f32], &[f32], f32);

#[derive(Debug, Clone, Copy)]
pub struct ScaledAdd(ScaledAddFn);

impl ScaledAdd {
    pub fn for_isa(isa: SimdIsa) -> Option<Self> {
        match isa {
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 if std::arch::is_x86_feature_detected!("avx2") => Some(Self(x86::avx2)),
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx512 if std::arch::is_x86_feature_detected!("avx512f") => {
                Some(Self(x86::avx512))
            }
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon if std::arch::is_aarch64_feature_detected!("neon") => {
                Some(Self(arm::neon))
            }
            _ => None,
        }
    }

    /// Add the scaled input to the output. The two slices must have equal lengths.
    pub fn apply(self, output: &mut [f32], input: &[f32], factor: f32) {
        assert_eq!(output.len(), input.len());
        // SAFETY: The constructor checks CPU support and the slices have equal lengths.
        unsafe { (self.0)(output, input, factor) }
    }
}

#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn scalar_tail(output: &mut [f32], input: &[f32], factor: f32) {
    for (out, &value) in output.iter_mut().zip(input) {
        *out += factor * value;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn avx2(output: &mut [f32], input: &[f32], factor: f32) {
        let (output, output_tail) = output.as_chunks_mut::<8>();
        let (input, input_tail) = input.as_chunks::<8>();
        let factors = _mm256_set1_ps(factor);
        for (out, values) in output.iter_mut().zip(input) {
            // SAFETY: Each input and output array has eight accessible elements.
            unsafe {
                let previous = _mm256_loadu_ps(out.as_ptr());
                let values = _mm256_loadu_ps(values.as_ptr());
                let sum = _mm256_add_ps(previous, _mm256_mul_ps(factors, values));
                _mm256_storeu_ps(out.as_mut_ptr(), sum);
            }
        }
        super::scalar_tail(output_tail, input_tail, factor);
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn avx512(output: &mut [f32], input: &[f32], factor: f32) {
        let (output, output_tail) = output.as_chunks_mut::<16>();
        let (input, input_tail) = input.as_chunks::<16>();
        let factors = _mm512_set1_ps(factor);
        for (out, values) in output.iter_mut().zip(input) {
            // SAFETY: Each input and output array has sixteen accessible elements.
            unsafe {
                let previous = _mm512_loadu_ps(out.as_ptr());
                let values = _mm512_loadu_ps(values.as_ptr());
                let sum = _mm512_add_ps(previous, _mm512_mul_ps(factors, values));
                _mm512_storeu_ps(out.as_mut_ptr(), sum);
            }
        }
        super::scalar_tail(output_tail, input_tail, factor);
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn neon(output: &mut [f32], input: &[f32], factor: f32) {
        let (output, output_tail) = output.as_chunks_mut::<4>();
        let (input, input_tail) = input.as_chunks::<4>();
        let factors = vdupq_n_f32(factor);
        for (out, values) in output.iter_mut().zip(input) {
            // SAFETY: Each input and output array has four accessible elements.
            unsafe {
                let previous = vld1q_f32(out.as_ptr());
                let values = vld1q_f32(values.as_ptr());
                let sum = vaddq_f32(previous, vmulq_f32(factors, values));
                vst1q_f32(out.as_mut_ptr(), sum);
            }
        }
        super::scalar_tail(output_tail, input_tail, factor);
    }
}

#[cfg(test)]
mod tests {
    use super::{ScaledAdd, SimdIsa};

    #[test]
    fn available_kernels_match_the_scalar_reference() {
        for isa in [SimdIsa::Avx2, SimdIsa::Avx512, SimdIsa::Neon] {
            let Some(kernel) = ScaledAdd::for_isa(isa) else {
                continue;
            };
            let input: Vec<_> = (0..38).map(|index| index as f32 / 7.0 - 3.0).collect();
            let mut output = [0.1; 39];
            let input = &input[1..];
            let values = &mut output[1..38];
            for factor in [1.123, -0.347] {
                kernel.apply(values, input, factor);
            }
            for (&actual, &value) in values.iter().zip(input) {
                let expected = (0.1 + value * 1.123) + value * -0.347;
                assert_eq!(actual.to_bits(), expected.to_bits(), "{isa:?}");
            }
            assert_eq!((output[0], output[38]), (0.1, 0.1));
            println!("verified {isa:?}");
        }
    }
}
