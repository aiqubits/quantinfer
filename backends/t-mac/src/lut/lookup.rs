#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LutIsa {
    Scalar,
    Avx2,
    Neon,
}

#[derive(Clone, Copy)]
pub(super) struct Selector {
    pub isa: LutIsa,
    apply: unsafe fn(&[f32; 16], u16) -> f32,
}

fn scalar(table: &[f32; 16], word: u16) -> f32 {
    table[(word & 15) as usize]
        + 2.0 * table[((word >> 4) & 15) as usize]
        + 4.0 * table[((word >> 8) & 15) as usize]
        - 8.0 * table[(word >> 12) as usize]
}

impl Selector {
    pub fn scalar() -> Self {
        Self {
            isa: LutIsa::Scalar,
            apply: scalar,
        }
    }
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            return Self {
                isa: LutIsa::Avx2,
                apply: avx2,
            };
        }
        #[cfg(all(target_arch = "aarch64", target_endian = "little"))]
        if std::arch::is_aarch64_feature_detected!("neon") {
            return Self {
                isa: LutIsa::Neon,
                apply: neon,
            };
        }
        Self::scalar()
    }
    pub fn dot(self, table: &[f32; 16], word: u16) -> f32 {
        // SAFETY: constructors bind only an ISA detected on this host. Every
        // four-bit table index is 0..15, within the 16-element borrowed table.
        unsafe { (self.apply)(table, word) }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2(table: &[f32; 16], word: u16) -> f32 {
    use std::arch::x86_64::*;
    // SAFETY: the caller checked AVX2. Four masked indices select valid f32
    // elements; gather scale 4 converts element indices to byte offsets.
    unsafe {
        let indices = _mm_set_epi32(
            (word >> 12) as i32,
            ((word >> 8) & 15) as i32,
            ((word >> 4) & 15) as i32,
            (word & 15) as i32,
        );
        let values = _mm_i32gather_ps::<4>(table.as_ptr(), indices);
        let values = _mm_mul_ps(values, _mm_set_ps(-8.0, 4.0, 2.0, 1.0));
        let sum = _mm_add_ps(values, _mm_movehl_ps(values, values));
        _mm_cvtss_f32(_mm_add_ss(sum, _mm_shuffle_ps::<0x55>(sum, sum)))
    }
}

#[cfg(all(target_arch = "aarch64", target_endian = "little"))]
#[target_feature(enable = "neon")]
unsafe fn neon(table: &[f32; 16], word: u16) -> f32 {
    use std::arch::aarch64::*;
    let mut indices = [0u8; 16];
    for lane in 0..4 {
        for byte in 0..4 {
            indices[lane * 4 + byte] = (((word >> (lane * 4)) & 15) * 4 + byte as u16) as u8;
        }
    }
    let coefficients = [1.0f32, 2.0, 4.0, -8.0];
    // SAFETY: all four loads together read exactly the 64-byte table. TBL
    // indices are 0..63 and reconstruct four valid f32 lanes on little-endian
    // AArch64. The caller checked NEON; stack arrays provide full load ranges.
    unsafe {
        let bytes = table.as_ptr().cast::<u8>();
        let tables = uint8x16x4_t(
            vld1q_u8(bytes),
            vld1q_u8(bytes.add(16)),
            vld1q_u8(bytes.add(32)),
            vld1q_u8(bytes.add(48)),
        );
        let values = vreinterpretq_f32_u8(vqtbl4q_u8(tables, vld1q_u8(indices.as_ptr())));
        vaddvq_f32(vmulq_f32(values, vld1q_f32(coefficients.as_ptr())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn every_bit_plane_index_matches_the_scalar_selector() {
        let selector = Selector::detect();
        let table = std::array::from_fn(|index| index as f32 * 0.25 - 1.75);
        for word in 0..=u16::MAX {
            assert_eq!(
                selector.dot(&table, word),
                scalar(&table, word),
                "word {word}"
            );
        }
        println!("verified T-MAC LUT4 {:?}", selector.isa);
    }
}
