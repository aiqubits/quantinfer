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

type Q8DotFn = unsafe fn(&[f32; 32], f32, &[i8; 32]) -> f32;
type Q4DotFn = unsafe fn(&[f32; 32], f32, &[u8; 32]) -> f32;

/// SIMD dot products for the block layouts used by GGML output-row weights.
/// The constructor performs runtime ISA detection; callers can keep the
/// scalar kernel as the fallback when an ISA is unavailable.
#[derive(Debug, Clone, Copy)]
pub struct QuantizedDot {
    q8: Q8DotFn,
    q4: Q4DotFn,
}

impl QuantizedDot {
    pub fn for_isa(isa: SimdIsa) -> Option<Self> {
        match isa {
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx2 if std::arch::is_x86_feature_detected!("avx2") => Some(Self {
                q8: x86::q8_avx2,
                q4: x86::q4_avx2,
            }),
            #[cfg(target_arch = "x86_64")]
            SimdIsa::Avx512 if std::arch::is_x86_feature_detected!("avx512f") => Some(Self {
                q8: x86::q8_avx512,
                q4: x86::q4_avx512,
            }),
            #[cfg(target_arch = "aarch64")]
            SimdIsa::Neon if std::arch::is_aarch64_feature_detected!("neon") => Some(Self {
                q8: arm::q8_neon,
                q4: arm::q4_neon,
            }),
            _ => None,
        }
    }

    pub fn q8(self, input: &[f32; 32], scale: f32, values: &[i8; 32]) -> f32 {
        // SAFETY: the constructor checks the target ISA and all block arrays
        // are fixed at 32 elements.
        unsafe { (self.q8)(input, scale, values) }
    }

    pub fn q4(self, input: &[f32; 32], scale: f32, values: &[u8; 32]) -> f32 {
        // SAFETY: the constructor checks the target ISA and all block arrays
        // are fixed at 32 elements.
        unsafe { (self.q4)(input, scale, values) }
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
    pub(super) unsafe fn q8_avx2(input: &[f32; 32], scale: f32, values: &[i8; 32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        for index in (0..32).step_by(8) {
            // SAFETY: each slice has 32 elements and index advances by eight.
            let lhs = unsafe { _mm256_loadu_ps(input.as_ptr().add(index)) };
            let codes = unsafe { _mm_loadl_epi64(values.as_ptr().add(index).cast()) };
            let codes = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(codes));
            sum = _mm256_add_ps(sum, _mm256_mul_ps(lhs, codes));
        }
        // SAFETY: this function is compiled with AVX2 and receives a valid
        // accumulator produced by the AVX2 loop above.
        unsafe { horizontal_sum_avx2(sum) * scale }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn q4_avx2(input: &[f32; 32], scale: f32, values: &[u8; 32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let offset = _mm256_set1_ps(8.0);
        for index in (0..32).step_by(8) {
            // SAFETY: each slice has 32 elements and index advances by eight.
            let lhs = unsafe { _mm256_loadu_ps(input.as_ptr().add(index)) };
            let codes = unsafe { _mm_loadl_epi64(values.as_ptr().add(index).cast()) };
            let codes = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(codes)), offset);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(lhs, codes));
        }
        // SAFETY: this function is compiled with AVX2 and receives a valid
        // accumulator produced by the AVX2 loop above.
        unsafe { horizontal_sum_avx2(sum) * scale }
    }

    #[inline]
    unsafe fn horizontal_sum_avx2(value: __m256) -> f32 {
        // SAFETY: the caller guarantees AVX2; each intrinsic operates on the
        // local register value and does not access memory.
        unsafe {
            let high = _mm256_extractf128_ps(value, 1);
            let low = _mm256_castps256_ps128(value);
            let sum = _mm_add_ps(low, high);
            let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            let sum = _mm_add_ss(sum, _mm_shuffle_ps(sum, sum, 1));
            _mm_cvtss_f32(sum)
        }
    }

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

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn q8_avx512(input: &[f32; 32], scale: f32, values: &[i8; 32]) -> f32 {
        let mut sum = _mm512_setzero_ps();
        for index in (0..32).step_by(16) {
            // SAFETY: each slice has 32 elements and index advances by sixteen.
            let lhs = unsafe { _mm512_loadu_ps(input.as_ptr().add(index)) };
            let codes = unsafe { _mm_loadu_si128(values.as_ptr().add(index).cast()) };
            let codes = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(codes));
            sum = _mm512_add_ps(sum, _mm512_mul_ps(lhs, codes));
        }
        _mm512_reduce_add_ps(sum) * scale
    }

    #[target_feature(enable = "avx512f")]
    pub(super) unsafe fn q4_avx512(input: &[f32; 32], scale: f32, values: &[u8; 32]) -> f32 {
        let mut sum = _mm512_setzero_ps();
        let offset = _mm512_set1_ps(8.0);
        for index in (0..32).step_by(16) {
            // SAFETY: each slice has 32 elements and index advances by sixteen.
            let lhs = unsafe { _mm512_loadu_ps(input.as_ptr().add(index)) };
            let codes = unsafe { _mm_loadu_si128(values.as_ptr().add(index).cast()) };
            let codes = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(codes)), offset);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(lhs, codes));
        }
        _mm512_reduce_add_ps(sum) * scale
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use std::arch::aarch64::*;

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn q8_neon(input: &[f32; 32], scale: f32, values: &[i8; 32]) -> f32 {
        let mut sum = vdupq_n_f32(0.0);
        for index in (0..32).step_by(8) {
            // SAFETY: each slice has 32 elements and index advances by eight.
            let lhs_low = unsafe { vld1q_f32(input.as_ptr().add(index)) };
            let lhs_high = unsafe { vld1q_f32(input.as_ptr().add(index + 4)) };
            let codes = unsafe { vld1_s8(values.as_ptr().add(index)) };
            let codes = vmovl_s8(codes);
            let low = vcvtq_f32_s32(vmovl_s16(vget_low_s16(codes)));
            let high = vcvtq_f32_s32(vmovl_s16(vget_high_s16(codes)));
            sum = vaddq_f32(sum, vmulq_f32(lhs_low, low));
            sum = vaddq_f32(sum, vmulq_f32(lhs_high, high));
        }
        (vaddvq_f32(sum)) * scale
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn q4_neon(input: &[f32; 32], scale: f32, values: &[u8; 32]) -> f32 {
        let mut sum = vdupq_n_f32(0.0);
        let offset = vdupq_n_f32(8.0);
        for index in (0..32).step_by(8) {
            // SAFETY: each slice has 32 elements and index advances by eight.
            let lhs_low = unsafe { vld1q_f32(input.as_ptr().add(index)) };
            let lhs_high = unsafe { vld1q_f32(input.as_ptr().add(index + 4)) };
            let codes = unsafe { vld1_u8(values.as_ptr().add(index)) };
            let codes = vmovl_u8(codes);
            let low = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(codes))), offset);
            let high = vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(codes))), offset);
            sum = vaddq_f32(sum, vmulq_f32(lhs_low, low));
            sum = vaddq_f32(sum, vmulq_f32(lhs_high, high));
        }
        (vaddvq_f32(sum)) * scale
    }

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
    use super::{QuantizedDot, ScaledAdd, SimdIsa};

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

    #[test]
    fn quantized_dot_matches_scalar_reference() {
        let input: [f32; 32] = std::array::from_fn(|index| index as f32 / 11.0 - 1.5);
        let q8: [i8; 32] = std::array::from_fn(|index| (index as i8 % 17) - 8);
        let q4: [u8; 32] = std::array::from_fn(|index| (index % 16) as u8);
        let expected_q8 = 0.37
            * input
                .iter()
                .zip(q8)
                .map(|(&left, right)| left * f32::from(right))
                .sum::<f32>();
        let expected_q4 = -0.23
            * input
                .iter()
                .zip(q4)
                .map(|(&left, right)| left * (f32::from(right) - 8.0))
                .sum::<f32>();
        for isa in [SimdIsa::Avx2, SimdIsa::Avx512, SimdIsa::Neon] {
            let Some(kernel) = QuantizedDot::for_isa(isa) else {
                continue;
            };
            assert!(
                (kernel.q8(&input, 0.37, &q8) - expected_q8).abs() < 1e-4,
                "{isa:?}"
            );
            assert!(
                (kernel.q4(&input, -0.23, &q4) - expected_q4).abs() < 1e-4,
                "{isa:?}"
            );
        }
    }
}
