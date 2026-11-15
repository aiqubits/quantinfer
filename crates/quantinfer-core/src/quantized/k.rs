//! Q5_K and Q6_K logical block representations.

use crate::{CoreError, Result, Shape};

use super::Q4_K_BLOCK_SIZE;

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
        shape.validate()?;
        let elements = shape.num_elements()?;
        if !elements.is_multiple_of(Q4_K_BLOCK_SIZE) || blocks.len() * Q4_K_BLOCK_SIZE != elements {
            return Err(CoreError::InvalidData(
                "Q5_K block count does not match tensor shape".to_owned(),
            ));
        }
        if blocks
            .iter()
            .any(|block| !block.scale.is_finite() || !block.min_scale.is_finite())
        {
            return Err(CoreError::InvalidData(
                "Q5_K block scales must be finite".to_owned(),
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
        shape.validate()?;
        let elements = shape.num_elements()?;
        if !elements.is_multiple_of(Q4_K_BLOCK_SIZE) || blocks.len() * Q4_K_BLOCK_SIZE != elements {
            return Err(CoreError::InvalidData(
                "Q6_K block count does not match tensor shape".to_owned(),
            ));
        }
        if blocks.iter().any(|block| !block.scale.is_finite()) {
            return Err(CoreError::InvalidData(
                "Q6_K block scales must be finite".to_owned(),
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
