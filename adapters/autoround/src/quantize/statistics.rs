use super::invalid;
use quantinfer_core::Tensor;
use quantinfer_quant::Result;

/// Exact finite token rows for projection reconstruction. The byte limit
/// applies to the retained F32 values.
#[derive(Debug, Clone)]
pub struct AutoRoundCalibration {
    channels: usize,
    maximum_bytes: usize,
    data: Vec<f32>,
}

impl AutoRoundCalibration {
    pub fn new(channels: usize, maximum_bytes: usize) -> Result<Self> {
        if channels == 0 || channels.checked_mul(4).is_none_or(|n| n > maximum_bytes) {
            return Err(invalid(
                "AutoRound statistics budget cannot hold a token row",
            ));
        }
        Ok(Self {
            channels,
            maximum_bytes,
            data: Vec::new(),
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }
    pub fn rows(&self) -> usize {
        self.data.len() / self.channels
    }

    pub fn observe_rows(&mut self, values: &Tensor) -> Result<()> {
        let &[rows, channels] = values.shape().dims() else {
            return Err(invalid("AutoRound observations need [token_rows,input]"));
        };
        if rows == 0 || channels != self.channels {
            return Err(invalid(
                "AutoRound observation dimensions disagree with the calibration",
            ));
        }
        let values = values.as_f32()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("AutoRound observations must be finite"));
        }
        if self
            .data
            .len()
            .checked_add(values.len())
            .and_then(|n| n.checked_mul(4))
            .is_none_or(|n| n > self.maximum_bytes)
        {
            return Err(invalid(
                "AutoRound observations exceed the statistics budget",
            ));
        }
        self.data
            .try_reserve_exact(values.len())
            .map_err(|_| invalid("cannot allocate AutoRound observations"))?;
        self.data.extend_from_slice(values);
        Ok(())
    }

    pub(super) fn values(&self) -> &[f32] {
        &self.data
    }
}
