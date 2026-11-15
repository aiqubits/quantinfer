//! AutoAWQ configuration.

use serde::Deserialize;

pub(super) const AWQ_BITS: usize = 4;
pub(super) const AWQ_PACK_FACTOR: usize = 8;
// Raw AWQ slots for logical output indices 0..7.
pub(super) const AWQ_RAW_SLOT: [usize; AWQ_PACK_FACTOR] = [0, 4, 1, 5, 2, 6, 3, 7];

/// Configuration needed to decode one AutoAWQ linear weight.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct AwqConfig {
    /// AutoAWQ currently uses four-bit words.
    pub bits: usize,
    /// Number of input rows sharing one scale/zero-point group.
    pub group_size: usize,
    /// Symmetric sources may omit `qzeros`; the midpoint code is used then.
    #[serde(default)]
    pub sym: bool,
    /// Offset applied to packed zero-point codes. AutoAWQ uses zero by default.
    #[serde(default)]
    pub zero_offset: i32,
}

impl AwqConfig {
    pub fn new(group_size: usize, sym: bool) -> Self {
        Self {
            bits: AWQ_BITS,
            group_size,
            sym,
            zero_offset: 0,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.bits != AWQ_BITS {
            return Err(format!(
                "AutoAWQ SafeTensors currently supports {} bits, got {}",
                AWQ_BITS, self.bits
            ));
        }
        if self.group_size == 0 {
            return Err("AWQ group_size must be greater than zero".to_owned());
        }
        Ok(())
    }
}
