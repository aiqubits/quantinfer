//! Backend-independent tensor, shape, device, and storage primitives.

use serde::{Deserialize, Serialize};

use crate::{CoreError, Result};

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

    pub(crate) fn validate(&self) -> Result<()> {
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
