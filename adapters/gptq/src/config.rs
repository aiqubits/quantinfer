//! GPTQ quantization configuration.

use serde::Deserialize;

/// Configuration needed to decode one GPTQ linear weight.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct GptqConfig {
    /// Number of bits in each packed weight code.
    pub bits: usize,
    /// Number of input rows sharing one scale/zero-point group.
    pub group_size: usize,
    /// Symmetric sources may omit `qzeros`; the midpoint code is used then.
    #[serde(default)]
    pub sym: bool,
    /// Stored GPTQ zero points are normally one less than the logical code.
    /// This remains explicit for checkpoints using a different convention.
    #[serde(default = "default_zero_offset")]
    pub zero_offset: i32,
}

impl GptqConfig {
    pub fn new(bits: usize, group_size: usize, sym: bool) -> Self {
        Self {
            bits,
            group_size,
            sym,
            zero_offset: default_zero_offset(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.bits, 2 | 4 | 8) {
            return Err(format!("GPTQ supports 2, 4, or 8 bits, got {}", self.bits));
        }
        if self.group_size == 0 {
            return Err("GPTQ group_size must be greater than zero".to_owned());
        }
        if self.bits > 32 || !32usize.is_multiple_of(self.bits) {
            return Err(format!(
                "GPTQ bits must divide a 32-bit packed word, got {}",
                self.bits
            ));
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

const fn default_zero_offset() -> i32 {
    1
}
