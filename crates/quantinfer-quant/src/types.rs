//! Calibration and quantization planning types.

use std::{collections::BTreeMap, fmt};

use quantinfer_core::{QuantizationScheme, Tensor};

use crate::Result;

pub struct CalibrationDataset {
    samples: Vec<Tensor>,
}

impl CalibrationDataset {
    pub fn new(samples: Vec<Tensor>) -> Result<Self> {
        if samples.is_empty() {
            return Err(QuantizationError::InvalidInput(
                "calibration dataset cannot be empty".to_owned(),
            ));
        }
        let shape = samples[0].shape().clone();
        if samples.iter().any(|sample| sample.shape() != &shape) {
            return Err(QuantizationError::InvalidInput(
                "calibration samples must have identical shapes".to_owned(),
            ));
        }
        Ok(Self { samples })
    }

    pub fn samples(&self) -> &[Tensor] {
        &self.samples
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImatrixStatistics {
    count: usize,
    sum_abs: Vec<f64>,
    sum_square: Vec<f64>,
}

impl ImatrixStatistics {
    pub fn new(channels: usize) -> Result<Self> {
        if channels == 0 {
            return Err(QuantizationError::InvalidInput(
                "imatrix statistics require at least one channel".to_owned(),
            ));
        }
        Ok(Self {
            count: 0,
            sum_abs: vec![0.0; channels],
            sum_square: vec![0.0; channels],
        })
    }

    pub fn from_dataset(dataset: &CalibrationDataset) -> Result<Self> {
        let elements = dataset.samples()[0]
            .shape()
            .num_elements()
            .map_err(QuantizationError::Core)?;
        let mut statistics = Self::new(elements)?;
        for sample in dataset.samples() {
            statistics.observe(sample)?;
        }
        Ok(statistics)
    }

    pub fn observe(&mut self, sample: &Tensor) -> Result<()> {
        let values = sample.as_f32()?;
        if values.len() != self.sum_abs.len() {
            return Err(QuantizationError::InvalidInput(
                "calibration sample shape does not match imatrix statistics".to_owned(),
            ));
        }
        self.accumulate(values, 1)
    }

    /// Count each token row as one observation of the input channels.
    pub fn observe_rows(&mut self, sample: &Tensor) -> Result<()> {
        match sample.shape().dims() {
            &[rows, channels] if channels == self.sum_abs.len() => {
                self.accumulate(sample.as_f32()?, rows)
            }
            _ => Err(QuantizationError::InvalidInput(
                "imatrix activations must have shape [tokens, channels]".to_owned(),
            )),
        }
    }

    fn accumulate(&mut self, values: &[f32], rows: usize) -> Result<()> {
        let count = self.count.checked_add(rows).ok_or_else(|| {
            QuantizationError::InvalidInput("imatrix observation count overflows usize".to_owned())
        })?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(QuantizationError::InvalidInput(
                "imatrix activations must be finite".to_owned(),
            ));
        }
        for row in values.chunks_exact(self.sum_abs.len()) {
            for (index, &value) in row.iter().enumerate() {
                let value = f64::from(value);
                self.sum_abs[index] += value.abs();
                self.sum_square[index] += value * value;
            }
        }
        self.count = count;
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn mean_abs(&self) -> Vec<f32> {
        let denominator = self.count.max(1) as f64;
        self.sum_abs
            .iter()
            .map(|value| (value / denominator) as f32)
            .collect()
    }

    pub fn mean_square(&self) -> Vec<f32> {
        let denominator = self.count.max(1) as f64;
        self.sum_square
            .iter()
            .map(|value| (value / denominator) as f32)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MixedPrecisionPlan {
    assignments: BTreeMap<String, QuantizationScheme>,
}

impl MixedPrecisionPlan {
    pub fn assign(&mut self, tensor_name: impl Into<String>, scheme: QuantizationScheme) {
        self.assignments.insert(tensor_name.into(), scheme);
    }

    pub fn scheme_for(&self, tensor_name: &str) -> Option<QuantizationScheme> {
        self.assignments.get(tensor_name).copied()
    }

    pub fn assignments(&self) -> &BTreeMap<String, QuantizationScheme> {
        &self.assignments
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationError {
    InvalidInput(String),
    Core(quantinfer_core::CoreError),
}

impl fmt::Display for QuantizationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput(message) => formatter.write_str(message),
            Self::Core(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for QuantizationError {}

impl From<quantinfer_core::CoreError> for QuantizationError {
    fn from(error: quantinfer_core::CoreError) -> Self {
        Self::Core(error)
    }
}
