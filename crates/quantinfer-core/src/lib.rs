//! Core tensor, layout, device, and quantized storage primitives.

#![forbid(unsafe_code)]

use std::fmt;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    Q8_0,
}

impl DType {
    pub const fn element_size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q8_0 => 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        if dims.iter().any(|dimension| *dimension == 0) {
            return Err(CoreError::InvalidShape(
                "tensor dimensions must be greater than zero".to_owned(),
            ));
        }
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

    pub fn num_elements(&self) -> usize {
        self.0.iter().product()
    }

    pub fn last_dim(&self) -> Option<usize> {
        self.0.last().copied()
    }

    fn validate(&self) -> Result<()> {
        if self.0.iter().any(|dimension| *dimension == 0) {
            return Err(CoreError::InvalidShape(
                "tensor dimensions must be greater than zero".to_owned(),
            ));
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
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
        if shape.num_elements() != data.len() {
            return Err(CoreError::InvalidData(format!(
                "shape contains {} elements but data contains {}",
                shape.num_elements(),
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
        Self::from_f32(shape.clone(), vec![0.0; shape.num_elements()])
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
        if shape.num_elements() != self.shape.num_elements() {
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
        if shape.num_elements() % Q8_0_BLOCK_SIZE != 0 {
            return Err(CoreError::InvalidData(
                "Q8_0 tensor element count must be a multiple of 32".to_owned(),
            ));
        }
        if blocks.len() * Q8_0_BLOCK_SIZE != shape.num_elements() {
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
        self.shape.num_elements()
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
}
