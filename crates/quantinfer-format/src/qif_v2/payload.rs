//! Incremental tensor hashing and finite-value validation across chunk boundaries.
use super::invalid;
use crate::Result;
use quantinfer_core::DType;
use sha2::{Digest, Sha256};

fn block_bytes(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        DType::F16 | DType::BF16 => 2,
        DType::Q8_0 => 34,
        DType::Q4_0 => 18,
        DType::Q4_K => 144,
        DType::Q5_K => 176,
        DType::Q6_K => 210,
    }
}

pub(super) struct PayloadVerifier<'a> {
    dtype: DType,
    name: &'a str,
    hash: Sha256,
    length: u64,
    tail: Vec<u8>,
}
impl<'a> PayloadVerifier<'a> {
    pub(super) fn new(dtype: DType, name: &'a str) -> Self {
        Self {
            dtype,
            name,
            hash: Sha256::new(),
            length: 0,
            tail: Vec::with_capacity(block_bytes(dtype)),
        }
    }
    pub(super) fn update(&mut self, mut bytes: &[u8]) -> Result<()> {
        self.hash.update(bytes);
        self.length = self
            .length
            .checked_add(bytes.len() as u64)
            .ok_or_else(|| invalid("tensor verification length overflows"))?;
        let stride = block_bytes(self.dtype);
        if !self.tail.is_empty() {
            let take = (stride - self.tail.len()).min(bytes.len());
            self.tail.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.tail.len() < stride {
                return Ok(());
            }
            validate_payload(self.dtype, &self.tail, self.name)?;
            self.tail.clear();
        }
        let complete = bytes.len() / stride * stride;
        validate_payload(self.dtype, &bytes[..complete], self.name)?;
        self.tail.extend_from_slice(&bytes[complete..]);
        Ok(())
    }
    pub(super) fn finish(self, length: u64, expected: &[u8; 32]) -> Result<()> {
        if self.length != length || !self.tail.is_empty() {
            return Err(invalid(format!(
                "tensor {} verification length mismatch",
                self.name
            )));
        }
        let digest: [u8; 32] = self.hash.finalize().into();
        if digest != *expected {
            return Err(invalid(format!("tensor {} checksum mismatch", self.name)));
        }
        Ok(())
    }
}

pub(super) fn validate_payload(dtype: DType, bytes: &[u8], name: &str) -> Result<()> {
    use half::{bf16, f16};
    if !bytes.len().is_multiple_of(block_bytes(dtype)) {
        return Err(invalid(format!(
            "tensor {name} has an incomplete storage block"
        )));
    }
    let finite = |value: f32| {
        if value.is_finite() {
            Ok(())
        } else {
            Err(invalid(format!("tensor {name} contains non-finite values")))
        }
    };
    match dtype {
        DType::F32 => {
            for chunk in bytes.as_chunks::<4>().0 {
                finite(f32::from_le_bytes(*chunk))?;
            }
        }
        DType::F16 => {
            for chunk in bytes.as_chunks::<2>().0 {
                finite(f16::from_le_bytes(*chunk).to_f32())?;
            }
        }
        DType::BF16 => {
            for chunk in bytes.as_chunks::<2>().0 {
                finite(bf16::from_le_bytes(*chunk).to_f32())?;
            }
        }
        DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K => {
            let (block_bytes, scale_offsets): (usize, &[usize]) = match dtype {
                DType::Q8_0 => (34, &[0]),
                DType::Q4_0 => (18, &[0]),
                DType::Q4_K => (144, &[0, 2]),
                DType::Q5_K => (176, &[0, 2]),
                DType::Q6_K => (210, &[208]),
                DType::F32 | DType::F16 | DType::BF16 => unreachable!(),
            };
            for block in bytes.chunks_exact(block_bytes) {
                for &offset in scale_offsets {
                    let scale =
                        f16::from_le_bytes(block[offset..offset + 2].try_into().unwrap()).to_f32();
                    finite(scale)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_and_checks_finite_values_across_arbitrary_block_boundaries() {
        for dtype in [
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::Q8_0,
            DType::Q4_0,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
        ] {
            let width = block_bytes(dtype);
            let valid = vec![0; 2 * width];
            let expected: [u8; 32] = Sha256::digest(&valid).into();
            for step in [1, 3, 17, 64, 211] {
                let mut verifier = PayloadVerifier::new(dtype, "weight");
                for part in valid.chunks(step) {
                    verifier.update(part).unwrap();
                }
                verifier.finish(valid.len() as u64, &expected).unwrap();
                let mut invalid = valid.clone();
                match dtype {
                    DType::F32 => invalid[..4].copy_from_slice(&f32::INFINITY.to_le_bytes()),
                    DType::BF16 => invalid[..2].copy_from_slice(&0x7f80u16.to_le_bytes()),
                    DType::Q6_K => invalid[208..210].copy_from_slice(&0x7c00u16.to_le_bytes()),
                    _ => invalid[..2].copy_from_slice(&0x7c00u16.to_le_bytes()),
                }
                let expected: [u8; 32] = Sha256::digest(&invalid).into();
                let mut verifier = PayloadVerifier::new(dtype, "weight");
                let update = invalid
                    .chunks(step)
                    .try_for_each(|part| verifier.update(part));
                let result = update.and_then(|_| verifier.finish(invalid.len() as u64, &expected));
                assert!(result.unwrap_err().to_string().contains("non-finite"));
            }
        }
    }
}
