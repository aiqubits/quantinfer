//! Q8_0 logical block representation.

use crate::{CoreError, Result, Shape};

use super::Q8_0_BLOCK_SIZE;

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
        shape.validate()?;
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
        if blocks.iter().any(|block| !block.scale.is_finite()) {
            return Err(CoreError::InvalidData(
                "Q8_0 block scales must be finite".to_owned(),
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
