//! Exact GGML Q4_0 packing and bounds checked before native SDK calls.
use half::f16;
use quantinfer_core::QuantizedTensor4;
use quantinfer_kernel::{KernelError, Result};

fn invalid(message: &str) -> KernelError {
    KernelError::InvalidInput(message.into())
}

pub(crate) fn packed_size(rows: usize, k: usize, block_bytes: usize) -> Result<usize> {
    if rows == 0 || k == 0 || !k.is_multiple_of(32) {
        return Err(invalid(
            "KleidiAI requires nonempty matrices and K divisible by 32",
        ));
    }
    rows.checked_add(3)
        .map(|rows| rows / 4 * 4)
        .and_then(|rows| rows.checked_mul(k / 32))
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| invalid("KleidiAI packed size overflows"))
}

pub(crate) fn validate_lhs(values: &[f32]) -> Result<()> {
    if values.is_empty() || !values.len().is_multiple_of(32) {
        return Err(invalid(
            "KleidiAI activations require complete 32-element blocks",
        ));
    }
    for block in values.as_chunks::<32>().0 {
        if block.iter().any(|value| !value.is_finite()) {
            return Err(invalid("KleidiAI activations must be finite"));
        }
        let max = block.iter().copied().map(f32::abs).fold(0.0, f32::max);
        if max == 0.0 {
            continue;
        }
        let scale = max / 127.0;
        let stored = f16::from_f32(scale).to_f32();
        if stored == 0.0 || !stored.is_finite() || !scale.recip().is_finite() {
            return Err(invalid(
                "KleidiAI activation scale is outside its F16 profile",
            ));
        }
    }
    Ok(())
}

pub(crate) fn ggml_bytes(rhs: &QuantizedTensor4) -> Result<Vec<u8>> {
    let [k, n] = rhs.shape().dims() else {
        return Err(invalid("KleidiAI RHS must be a matrix"));
    };
    packed_size(*n, *k, 18)?;
    let mut bytes = Vec::new();
    let size = rhs
        .blocks()
        .len()
        .checked_mul(18)
        .ok_or_else(|| invalid("KleidiAI RHS size overflows"))?;
    bytes
        .try_reserve_exact(size)
        .map_err(|_| invalid("cannot allocate KleidiAI RHS"))?;
    for block in rhs.blocks() {
        let scale = f16::from_f32(block.scale);
        if !block.scale.is_finite()
            || scale.to_f32().to_bits() != block.scale.to_bits()
            || block.values.iter().any(|&value| value > 15)
        {
            return Err(invalid(
                "KleidiAI requires valid Q4_0 codes and exactly representable F16 weight scales",
            ));
        }
        bytes.extend_from_slice(&scale.to_bits().to_le_bytes());
        for index in 0..16 {
            bytes.push(block.values[index] | (block.values[index + 16] << 4));
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantinfer_core::Q4_0Block;
    #[test]
    fn exact_weight_packing_and_native_numeric_bounds() {
        let block = Q4_0Block {
            scale: -0.25,
            values: std::array::from_fn(|index| ((index * 7 + index / 4) % 16) as u8),
        };
        let rhs = QuantizedTensor4::from_blocks(vec![32, 1], vec![block.clone()]).unwrap();
        let bytes = ggml_bytes(&rhs).unwrap();
        assert_eq!(bytes.len(), 18);
        assert_eq!(
            f16::from_bits(u16::from_le_bytes(bytes[..2].try_into().unwrap())).to_f32(),
            -0.25
        );
        for index in 0..32 {
            let code = if index < 16 {
                bytes[index + 2] & 15
            } else {
                bytes[index - 16 + 2] >> 4
            };
            assert_eq!(code, block.values[index]);
        }
        let mut incompatible = rhs.clone();
        incompatible.blocks_mut()[0].scale = 0.1;
        assert!(ggml_bytes(&incompatible).is_err());
        assert!(packed_size(usize::MAX, 32, 18).is_err());
        assert!(packed_size(1, 33, 18).is_err());
        assert_eq!(packed_size(5, 64, 18).unwrap(), 8 * 2 * 18);
        assert_eq!(packed_size(3, 32, 34).unwrap(), 4 * 34);
        validate_lhs(&[0.0; 32]).unwrap();
        validate_lhs(&[1.0; 32]).unwrap();
        for value in [f32::NAN, f32::INFINITY, f32::MAX, f32::MIN_POSITIVE] {
            assert!(validate_lhs(&[value; 32]).is_err());
        }
    }
}
