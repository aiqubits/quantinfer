//! Shared projection encodings for native GPU kernels.

#![forbid(unsafe_code)]

use std::borrow::Cow;
use std::collections::BTreeSet;

use quantinfer_core::{
    DType, Device, QuantizationScheme, QuantizedTensor, QuantizedTensor4, Shape, Tensor,
    GGML_OUTPUT_ROWS_LAYOUT,
};
use quantinfer_kernel::{KernelError, Result};
use quantinfer_runtime::{KernelCapability, Operation};

/// Shader weights as little-endian words. Quantized blocks retain their F32
/// scales and codes. Q4_0 codes occupy consecutive nibbles in lane order.
pub enum ProjectionWeights<'a> {
    F32(&'a Tensor),
    Q4(&'a QuantizedTensor4),
    Q8(&'a QuantizedTensor),
}

impl<'a> ProjectionWeights<'a> {
    pub fn shape(&self) -> &Shape {
        match self {
            Self::F32(tensor) => tensor.shape(),
            Self::Q4(tensor) => tensor.shape(),
            Self::Q8(tensor) => tensor.shape(),
        }
    }

    pub fn encoding(&self) -> u32 {
        match self {
            Self::F32(_) => 0,
            Self::Q4(_) => 1,
            Self::Q8(_) => 2,
        }
    }

    pub fn byte_len(&self) -> Result<u64> {
        let (count, bytes) = match self {
            Self::F32(tensor) => (tensor.as_f32()?.len(), 4),
            Self::Q4(tensor) => (tensor.blocks().len(), 20),
            Self::Q8(tensor) => (tensor.blocks().len(), 36),
        };
        Ok((count as u64) * bytes)
    }

    pub fn bytes(self) -> Result<Cow<'a, [u8]>> {
        match self {
            Self::F32(tensor) => Ok(Cow::Borrowed(bytemuck::cast_slice(tensor.as_f32()?))),
            Self::Q4(tensor) => {
                let mut bytes = Vec::with_capacity(tensor.blocks().len() * 20);
                for block in tensor.blocks() {
                    if block.values.iter().any(|&code| code > 15) {
                        return Err(KernelError::InvalidInput(
                            "Q4_0 codes must fit in four bits".to_owned(),
                        ));
                    }
                    bytes.extend_from_slice(&block.scale.to_le_bytes());
                    bytes.extend(
                        block
                            .values
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .map(|[low, high]| low | (high << 4)),
                    );
                }
                Ok(Cow::Owned(bytes))
            }
            Self::Q8(tensor) => {
                let mut bytes = Vec::with_capacity(tensor.blocks().len() * 36);
                for block in tensor.blocks() {
                    bytes.extend_from_slice(&block.scale.to_le_bytes());
                    bytes.extend(block.values.map(|code| code as u8));
                }
                Ok(Cow::Owned(bytes))
            }
        }
    }
}

pub fn projection_capabilities() -> Vec<KernelCapability> {
    let capability = |dtype, quantization, layout: &str, operation, alignment| KernelCapability {
        device: Device::Gpu,
        dtype,
        quantization,
        tensor_layout: layout.to_owned(),
        supported_operations: BTreeSet::from([operation]),
        alignment,
        head_dimension: None,
        maximum_sequence_length: None,
    };
    let mut capabilities = vec![capability(
        DType::F32,
        QuantizationScheme::None,
        "row_major",
        Operation::Matmul,
        1,
    )];
    for (dtype, scheme) in [
        (DType::Q4_0, QuantizationScheme::Q4_0),
        (DType::Q8_0, QuantizationScheme::Q8_0),
    ] {
        for layout in ["row_major", GGML_OUTPUT_ROWS_LAYOUT] {
            capabilities.push(capability(
                dtype,
                scheme,
                layout,
                Operation::QuantizedMatmul,
                32,
            ));
        }
    }
    capabilities
}
