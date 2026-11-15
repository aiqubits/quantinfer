//! Q4_0 and Q4_K logical block representations.

use crate::{CoreError, Result, Shape};

use super::{Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE};

#[derive(Debug, Clone, PartialEq)]
pub struct Q4_0Block {
    pub scale: f32,
    /// Unpacked four-bit values in the range 0..=15. A format writer packs two
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
        shape.validate()?;
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
        if blocks.iter().any(|block| !block.scale.is_finite()) {
            return Err(CoreError::InvalidData(
                "Q4_0 block scales must be finite".to_owned(),
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
        shape.validate()?;
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
        if blocks
            .iter()
            .any(|block| !block.scale.is_finite() || !block.min_scale.is_finite())
        {
            return Err(CoreError::InvalidData(
                "Q4_K block scales must be finite".to_owned(),
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
