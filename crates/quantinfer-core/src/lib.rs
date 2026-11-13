//! Core tensor, layout, device, and quantized storage primitives.

#![forbid(unsafe_code)]

use std::fmt;

use serde::{Deserialize, Serialize};

pub type Result<T> = std::result::Result<T, CoreError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    InvalidShape(String),
    InvalidData(String),
    DTypeMismatch { expected: DType, actual: DType },
    Unsupported(String),
}

impl fmt::Display for CoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape(message)
            | Self::InvalidData(message)
            | Self::Unsupported(message) => formatter.write_str(message),
            Self::DTypeMismatch { expected, actual } => {
                write!(
                    formatter,
                    "dtype mismatch: expected {expected:?}, got {actual:?}"
                )
            }
        }
    }
}

impl std::error::Error for CoreError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q4_0,
    Q4_K,
    Q5_K,
    Q6_K,
}

impl DType {
    pub const fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q8_0 | Self::Q4_0 | Self::Q4_K | Self::Q5_K | Self::Q6_K => 1,
        }
    }

    pub const fn bit_width(self) -> usize {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::Q8_0 => 8,
            Self::Q4_0 => 4,
            Self::Q4_K => 4,
            Self::Q5_K => 5,
            Self::Q6_K => 6,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.contains(&0) {
            return Err(CoreError::InvalidShape(
                "tensor dimensions must be greater than zero".to_owned(),
            ));
        }
        dims.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| CoreError::InvalidShape("tensor shape overflows usize".to_owned()))
        })?;
        Ok(Self(dims))
    }

    pub fn scalar() -> Self {
        Self(Vec::new())
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn num_elements(&self) -> Result<usize> {
        self.0.iter().try_fold(1usize, |product, dimension| {
            product
                .checked_mul(*dimension)
                .ok_or_else(|| CoreError::InvalidShape("tensor shape overflows usize".to_owned()))
        })
    }

    pub fn last_dim(&self) -> Option<usize> {
        self.0.last().copied()
    }

    fn validate(&self) -> Result<()> {
        if self.0.contains(&0) {
            return Err(CoreError::InvalidShape(
                "tensor dimensions must be greater than zero".to_owned(),
            ));
        }
        self.num_elements()?;
        Ok(())
    }
}

impl From<usize> for Shape {
    fn from(value: usize) -> Self {
        Self(vec![value])
    }
}

impl From<Vec<usize>> for Shape {
    fn from(value: Vec<usize>) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Strides(Vec<usize>);

impl Strides {
    pub fn row_major(shape: &Shape) -> Self {
        let mut strides = vec![1; shape.rank()];
        for index in (0..shape.rank().saturating_sub(1)).rev() {
            strides[index] = strides[index + 1] * shape.dims()[index + 1];
        }
        Self(strides)
    }

    pub fn as_slice(&self) -> &[usize] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    /// An accelerator selected by a runtime provider.
    Gpu,
    /// A neural processing unit selected by a runtime provider.
    Npu,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Storage {
    F32(Vec<f32>),
    U16(Vec<u16>),
    U8(Vec<u8>),
}

impl Storage {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::U8(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            Self::F32(values) => Some(values),
            _ => None,
        }
    }

    pub fn as_f32_mut(&mut self) -> Option<&mut [f32]> {
        match self {
            Self::F32(values) => Some(values),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    storage: Storage,
    dtype: DType,
    shape: Shape,
    strides: Strides,
    device: Device,
}

impl Tensor {
    pub fn from_f32(shape: impl Into<Shape>, data: Vec<f32>) -> Result<Self> {
        let shape = shape.into();
        shape.validate()?;
        let elements = shape.num_elements()?;
        if elements != data.len() {
            return Err(CoreError::InvalidData(format!(
                "shape contains {elements} elements but data contains {}",
                data.len()
            )));
        }
        Ok(Self {
            strides: Strides::row_major(&shape),
            storage: Storage::F32(data),
            dtype: DType::F32,
            shape,
            device: Device::Cpu,
        })
    }

    pub fn zeros_f32(shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let elements = shape.num_elements()?;
        Self::from_f32(shape, vec![0.0; elements])
    }

    pub fn scalar(value: f32) -> Self {
        Self {
            storage: Storage::F32(vec![value]),
            dtype: DType::F32,
            shape: Shape::scalar(),
            strides: Strides(Vec::new()),
            device: Device::Cpu,
        }
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn as_f32(&self) -> Result<&[f32]> {
        self.storage.as_f32().ok_or(CoreError::DTypeMismatch {
            expected: DType::F32,
            actual: self.dtype,
        })
    }

    pub fn as_f32_mut(&mut self) -> Result<&mut [f32]> {
        self.storage.as_f32_mut().ok_or(CoreError::DTypeMismatch {
            expected: DType::F32,
            actual: self.dtype,
        })
    }

    pub fn reshape(&self, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        shape.validate()?;
        if shape.num_elements()? != self.shape.num_elements()? {
            return Err(CoreError::InvalidShape(
                "reshape must preserve the number of elements".to_owned(),
            ));
        }
        Ok(Self {
            storage: self.storage.clone(),
            dtype: self.dtype,
            strides: Strides::row_major(&shape),
            shape,
            device: self.device,
        })
    }

    pub fn view(&self) -> TensorView<'_> {
        TensorView { tensor: self }
    }

    pub fn get_f32(&self, indices: &[usize]) -> Result<f32> {
        if indices.len() != self.shape.rank() {
            return Err(CoreError::InvalidShape(
                "index rank does not match tensor rank".to_owned(),
            ));
        }
        let mut offset = 0;
        for ((index, dimension), stride) in indices
            .iter()
            .zip(self.shape.dims())
            .zip(self.strides.as_slice())
        {
            if *index >= *dimension {
                return Err(CoreError::InvalidShape(
                    "tensor index is out of bounds".to_owned(),
                ));
            }
            offset += *index * *stride;
        }
        Ok(self.as_f32()?[offset])
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    tensor: &'a Tensor,
}

impl<'a> TensorView<'a> {
    pub fn dtype(self) -> DType {
        self.tensor.dtype()
    }

    pub fn shape(self) -> &'a Shape {
        self.tensor.shape()
    }

    pub fn strides(self) -> &'a Strides {
        self.tensor.strides()
    }

    pub fn as_f32(self) -> Result<&'a [f32]> {
        self.tensor.as_f32()
    }

    pub fn tensor(self) -> &'a Tensor {
        self.tensor
    }
}

pub const Q8_0_BLOCK_SIZE: usize = 32;
pub const GGML_OUTPUT_ROWS_LAYOUT: &str = "ggml_output_rows";

#[derive(Debug, Clone, PartialEq)]
pub struct Q8_0Block {
    pub scale: f32,
    pub values: [i8; Q8_0_BLOCK_SIZE],
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor {
    shape: Shape,
    blocks: Vec<Q8_0Block>,
}

impl QuantizedTensor {
    pub fn from_blocks(shape: impl Into<Shape>, blocks: Vec<Q8_0Block>) -> Result<Self> {
        let shape = shape.into();
        let elements = shape.num_elements()?;
        if !elements.is_multiple_of(Q8_0_BLOCK_SIZE) {
            return Err(CoreError::InvalidData(
                "Q8_0 tensor element count must be a multiple of 32".to_owned(),
            ));
        }
        if blocks.len() * Q8_0_BLOCK_SIZE != elements {
            return Err(CoreError::InvalidData(
                "Q8_0 block count does not match tensor shape".to_owned(),
            ));
        }
        Ok(Self { shape, blocks })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn blocks(&self) -> &[Q8_0Block] {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut [Q8_0Block] {
        &mut self.blocks
    }

    pub fn num_elements(&self) -> usize {
        self.shape
            .num_elements()
            .expect("quantized tensor shape is validated at construction")
    }
}

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

pub const Q4_0_BLOCK_SIZE: usize = 32;
pub const Q4_K_BLOCK_SIZE: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct Q4_0Block {
    pub scale: f32,
    /// Unpacked values in the GGML range 0..=15. The format writer packs two
    /// values per byte, while kernels consume this logical representation.
    pub values: [u8; Q4_0_BLOCK_SIZE],
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensor4 {
    shape: Shape,
    blocks: Vec<Q4_0Block>,
}

impl QuantizedTensor4 {
    pub fn from_blocks(shape: impl Into<Shape>, blocks: Vec<Q4_0Block>) -> Result<Self> {
        let shape = shape.into();
        let elements = shape.num_elements()?;
        if !elements.is_multiple_of(Q4_0_BLOCK_SIZE) {
            return Err(CoreError::InvalidData(
                "Q4_0 tensor element count must be a multiple of 32".to_owned(),
            ));
        }
        if blocks.len() * Q4_0_BLOCK_SIZE != elements {
            return Err(CoreError::InvalidData(
                "Q4_0 block count does not match tensor shape".to_owned(),
            ));
        }
        if blocks
            .iter()
            .flat_map(|block| block.values)
            .any(|value| value > 15)
        {
            return Err(CoreError::InvalidData(
                "Q4_0 values must fit in four bits".to_owned(),
            ));
        }
        Ok(Self { shape, blocks })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn blocks(&self) -> &[Q4_0Block] {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut [Q4_0Block] {
        &mut self.blocks
    }

    pub fn num_elements(&self) -> usize {
        self.shape
            .num_elements()
            .expect("quantized tensor shape is validated at construction")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q4KBlock {
    pub scale: f32,
    pub min_scale: f32,
    pub scales: [u8; 12],
    pub values: [u8; Q4_K_BLOCK_SIZE / 2],
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensorQ4K {
    shape: Shape,
    blocks: Vec<Q4KBlock>,
}

impl QuantizedTensorQ4K {
    pub fn from_blocks(shape: impl Into<Shape>, blocks: Vec<Q4KBlock>) -> Result<Self> {
        let shape = shape.into();
        let elements = shape.num_elements()?;
        if !elements.is_multiple_of(Q4_K_BLOCK_SIZE) {
            return Err(CoreError::InvalidData(
                "Q4_K tensor element count must be a multiple of 256".to_owned(),
            ));
        }
        if blocks.len() * Q4_K_BLOCK_SIZE != elements {
            return Err(CoreError::InvalidData(
                "Q4_K block count does not match tensor shape".to_owned(),
            ));
        }
        if blocks
            .iter()
            .flat_map(|block| block.values)
            .flat_map(|value| [value & 0x0f, value >> 4])
            .any(|value| value > 15)
        {
            return Err(CoreError::InvalidData(
                "Q4_K values must fit in four bits".to_owned(),
            ));
        }
        Ok(Self { shape, blocks })
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn blocks(&self) -> &[Q4KBlock] {
        &self.blocks
    }

    pub fn blocks_mut(&mut self) -> &mut [Q4KBlock] {
        &mut self.blocks
    }

    pub fn num_elements(&self) -> usize {
        self.shape
            .num_elements()
            .expect("quantized tensor shape is validated at construction")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q5KBlock {
    pub scale: f32,
    pub min_scale: f32,
    pub scales: [u8; 12],
    pub high_bits: [u8; 32],
    pub values: [u8; 128],
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensorQ5K {
    shape: Shape,
    blocks: Vec<Q5KBlock>,
}

impl QuantizedTensorQ5K {
    pub fn from_blocks(shape: impl Into<Shape>, blocks: Vec<Q5KBlock>) -> Result<Self> {
        let shape = shape.into();
        let elements = shape.num_elements()?;
        if !elements.is_multiple_of(Q4_K_BLOCK_SIZE) || blocks.len() * Q4_K_BLOCK_SIZE != elements {
            return Err(CoreError::InvalidData(
                "Q5_K block count does not match tensor shape".to_owned(),
            ));
        }
        Ok(Self { shape, blocks })
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn blocks(&self) -> &[Q5KBlock] {
        &self.blocks
    }
    pub fn num_elements(&self) -> usize {
        self.shape
            .num_elements()
            .expect("quantized tensor shape is validated at construction")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Q6KBlock {
    pub scale: f32,
    pub scales: [i8; 16],
    pub low_bits: [u8; 128],
    pub high_bits: [u8; 64],
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedTensorQ6K {
    shape: Shape,
    blocks: Vec<Q6KBlock>,
}

impl QuantizedTensorQ6K {
    pub fn from_blocks(shape: impl Into<Shape>, blocks: Vec<Q6KBlock>) -> Result<Self> {
        let shape = shape.into();
        let elements = shape.num_elements()?;
        if !elements.is_multiple_of(Q4_K_BLOCK_SIZE) || blocks.len() * Q4_K_BLOCK_SIZE != elements {
            return Err(CoreError::InvalidData(
                "Q6_K block count does not match tensor shape".to_owned(),
            ));
        }
        Ok(Self { shape, blocks })
    }
    pub fn shape(&self) -> &Shape {
        &self.shape
    }
    pub fn blocks(&self) -> &[Q6KBlock] {
        &self.blocks
    }
    pub fn num_elements(&self) -> usize {
        self.shape
            .num_elements()
            .expect("quantized tensor shape is validated at construction")
    }
}

#[cfg(test)]
mod tests {
    use super::{DType, Shape, Tensor};

    #[test]
    fn creates_and_indexes_contiguous_tensor() {
        let tensor = Tensor::from_f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.get_f32(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn reshape_preserves_values() {
        let tensor = Tensor::from_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let reshaped = tensor.reshape(Shape::new(vec![4]).unwrap()).unwrap();
        assert_eq!(reshaped.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn validates_quantized_storage_descriptor() {
        let descriptor = super::TensorStorageDescriptor::quantized(
            DType::Q8_0,
            super::QuantizationScheme::Q8_0,
            Shape::from(vec![64]),
            32,
        )
        .unwrap();
        descriptor.validate().unwrap();
        assert_eq!(descriptor.dtype.bit_width(), 8);
        assert!(super::TensorStorageDescriptor::quantized(
            DType::Q4_0,
            super::QuantizationScheme::Q4_0,
            Shape::from(vec![31]),
            32,
        )
        .is_err());
        let mut invalid = descriptor;
        invalid.quantization = super::QuantizationScheme::Q4_0;
        assert!(invalid.validate().is_err());
        for (dtype, scheme) in [
            (DType::Q5_K, super::QuantizationScheme::Q5_K),
            (DType::Q6_K, super::QuantizationScheme::Q6_K),
        ] {
            let descriptor = super::TensorStorageDescriptor::quantized(
                dtype,
                scheme,
                Shape::from(vec![2, 256]),
                super::Q4_K_BLOCK_SIZE,
            )
            .unwrap();
            assert_eq!(descriptor.quantization, scheme);
        }
    }

    #[test]
    fn rejects_overflowing_shapes() {
        assert!(Shape::new(vec![usize::MAX, 2]).is_err());
        assert!(Tensor::from_f32(Shape::from(vec![usize::MAX, 2]), Vec::new()).is_err());
    }
}
