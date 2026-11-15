//! Compressed-tensors WNA16 configuration.

use serde::Deserialize;

/// Configuration for one compressed-tensors WNA16 matrix.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct LlmCompressorConfig {
    pub bits: usize,
    pub group_size: usize,
    #[serde(default)]
    pub symmetric: bool,
    /// Some exporters store a zero point with a fixed integer offset.
    #[serde(default)]
    pub zero_offset: i32,
}

impl LlmCompressorConfig {
    pub fn new(bits: usize, group_size: usize, symmetric: bool) -> Self {
        Self {
            bits,
            group_size,
            symmetric,
            zero_offset: 0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.bits, 2 | 4 | 8) {
            return Err(format!(
                "compressed-tensors supports 2, 4, or 8 bits, got {}",
                self.bits
            ));
        }
        if self.group_size == 0 {
            return Err("compressed-tensors group_size must be greater than zero".to_owned());
        }
        Ok(())
    }

    pub(super) fn pack_factor(&self) -> usize {
        32 / self.bits
    }

    pub(super) fn code_mask(&self) -> u32 {
        (1_u32 << self.bits) - 1
    }
}
