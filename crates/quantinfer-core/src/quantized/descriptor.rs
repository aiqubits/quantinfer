//! Logical storage descriptors and quantization schemes.

use serde::{Deserialize, Serialize};

use crate::{CoreError, DType, Result, Shape};

/// The logical quantization scheme independent of a file format or backend
/// packing. It is used by model manifests and execution capability matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantizationScheme {
    None,
    Q8_0,
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
    /// A model-level variant that combines different tensor schemes.
    Mixed,
}

impl QuantizationScheme {
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "f32",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4_K => "q4_k",
            Self::Q5_K => "q5_k",
            Self::Q6_K => "q6_k",
            Self::Mixed => "mixed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageType {
    F32,
    F16,
    BF16,
    Quantized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScaleType {
    F32,
    F16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ZeroPointType {
    None,
    U8,
    I8,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationAxis {
    None,
    Tensor,
    Axis(usize),
}

/// Describes how a logical tensor is stored without prescribing a backend's
/// physical implementation details.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorStorageDescriptor {
    pub storage_type: StorageType,
    pub dtype: DType,
    pub quantization: QuantizationScheme,
    pub block_size: Option<usize>,
    pub group_size: Option<usize>,
    pub scale_type: Option<ScaleType>,
    pub zero_point_type: ZeroPointType,
    pub quantization_axis: QuantizationAxis,
    pub logical_shape: Shape,
    pub physical_layout: String,
    pub packing: String,
    pub backend_requirements: Vec<String>,
}

pub type QuantizedTensorDescriptor = TensorStorageDescriptor;

impl TensorStorageDescriptor {
    pub fn floating(dtype: DType, logical_shape: Shape) -> Result<Self> {
        let storage_type = match dtype {
            DType::F32 => StorageType::F32,
            DType::F16 => StorageType::F16,
            DType::BF16 => StorageType::BF16,
            DType::Q8_0 | DType::Q4_0 | DType::Q4_K | DType::Q5_K | DType::Q6_K => {
                return Err(CoreError::InvalidData(
                    "floating descriptor requires F32, F16, or BF16".to_owned(),
                ))
            }
        };
        Ok(Self {
            storage_type,
            dtype,
            quantization: QuantizationScheme::None,
            block_size: None,
            group_size: None,
            scale_type: None,
            zero_point_type: ZeroPointType::None,
            quantization_axis: QuantizationAxis::None,
            logical_shape,
            physical_layout: "row_major".to_owned(),
            packing: "none".to_owned(),
            backend_requirements: Vec::new(),
        })
    }

    pub fn quantized(
        dtype: DType,
        quantization: QuantizationScheme,
        logical_shape: Shape,
        block_size: usize,
    ) -> Result<Self> {
        let descriptor = Self {
            storage_type: StorageType::Quantized,
            dtype,
            quantization,
            block_size: Some(block_size),
            group_size: None,
            scale_type: Some(ScaleType::F16),
            zero_point_type: ZeroPointType::None,
            quantization_axis: QuantizationAxis::Tensor,
            logical_shape,
            physical_layout: "row_major".to_owned(),
            packing: match quantization {
                QuantizationScheme::Q8_0 => "scalar".to_owned(),
                QuantizationScheme::Q4_0 => "nibble_split_half".to_owned(),
                QuantizationScheme::Q4_K => "k_quant".to_owned(),
                QuantizationScheme::Q5_K => "k_quant".to_owned(),
                QuantizationScheme::Q6_K => "k_quant".to_owned(),
                QuantizationScheme::None | QuantizationScheme::Mixed => "none".to_owned(),
            },
            backend_requirements: Vec::new(),
        };
        descriptor.validate()?;
        Ok(descriptor)
    }

    pub fn validate(&self) -> Result<()> {
        if matches!(self.quantization_axis, QuantizationAxis::Axis(axis)
            if axis >= self.logical_shape.dims().len())
            || self.group_size == Some(0)
        {
            return Err(CoreError::InvalidData(
                "quantization axis or group size is invalid".to_owned(),
            ));
        }
        if self.logical_shape.dims().contains(&0) {
            return Err(CoreError::InvalidShape(
                "storage descriptor dimensions must be greater than zero".to_owned(),
            ));
        }
        let elements = self
            .logical_shape
            .dims()
            .iter()
            .try_fold(1usize, |product, dimension| product.checked_mul(*dimension))
            .ok_or_else(|| {
                CoreError::InvalidShape("storage descriptor shape overflows usize".to_owned())
            })?;
        let valid = match (self.storage_type, self.dtype, self.quantization) {
            (StorageType::F32, DType::F32, QuantizationScheme::None)
            | (StorageType::F16, DType::F16, QuantizationScheme::None)
            | (StorageType::BF16, DType::BF16, QuantizationScheme::None) => {
                self.block_size.is_none()
                    && self.scale_type.is_none()
                    && self.zero_point_type == ZeroPointType::None
                    && matches!(self.quantization_axis, QuantizationAxis::None)
            }
            (StorageType::Quantized, DType::Q8_0, QuantizationScheme::Q8_0)
            | (StorageType::Quantized, DType::Q4_0, QuantizationScheme::Q4_0)
            | (StorageType::Quantized, DType::Q4_K, QuantizationScheme::Q4_K)
            | (StorageType::Quantized, DType::Q5_K, QuantizationScheme::Q5_K)
            | (StorageType::Quantized, DType::Q6_K, QuantizationScheme::Q6_K) => {
                self.block_size
                    .is_some_and(|size| size > 0 && elements.is_multiple_of(size))
                    && self.scale_type.is_some()
                    && !matches!(self.quantization_axis, QuantizationAxis::None)
            }
            _ => false,
        };
        if valid {
            Ok(())
        } else {
            Err(CoreError::InvalidData(
                "storage type, dtype, and quantization scheme are inconsistent".to_owned(),
            ))
        }
    }
}
