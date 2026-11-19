use super::invalid;
use quantinfer_core::Tensor;
use quantinfer_quant::Result;

/// Mean absolute input activations; each actual token row contributes once.
#[derive(Debug, Clone, PartialEq)]
pub struct AwqCalibration {
    sums: Vec<f64>,
    rows: usize,
}
impl AwqCalibration {
    pub fn new(channels: usize, maximum_bytes: usize) -> Result<Self> {
        if channels == 0 || channels.checked_mul(8).is_none_or(|n| n > maximum_bytes) {
            return Err(invalid(
                "AWQ statistics exceed the budget or have zero channels",
            ));
        }
        let mut sums = Vec::new();
        sums.try_reserve_exact(channels)
            .map_err(|_| invalid("cannot allocate AWQ statistics"))?;
        sums.resize(channels, 0.);
        Ok(Self { sums, rows: 0 })
    }
    pub fn channels(&self) -> usize {
        self.sums.len()
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn observe_rows(&mut self, input: &Tensor) -> Result<()> {
        let &[rows, channels] = input.shape().dims() else {
            return Err(invalid("AWQ observations must be [tokens,channels]"));
        };
        if channels != self.channels() || rows == 0 {
            return Err(invalid("AWQ observation shape disagrees with statistics"));
        }
        let total = self
            .rows
            .checked_add(rows)
            .ok_or_else(|| invalid("AWQ observation count overflows"))?;
        let values = input.as_f32()?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(invalid("AWQ observations must be finite"));
        }
        for row in values.chunks_exact(channels) {
            for (sum, &v) in self.sums.iter_mut().zip(row) {
                *sum += f64::from(v).abs();
            }
        }
        self.rows = total;
        Ok(())
    }
    pub(super) fn mean_absolute(&self) -> Result<Vec<f64>> {
        if self.rows == 0 || self.sums.iter().all(|x| *x == 0.) {
            return Err(invalid("AWQ needs nonzero observed activation statistics"));
        }
        let mut means = super::buffer(self.sums.len())?;
        means.extend(self.sums.iter().map(|x| x / self.rows as f64));
        Ok(means)
    }
}
