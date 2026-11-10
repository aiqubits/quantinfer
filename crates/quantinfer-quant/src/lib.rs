//! Weight-only quantization primitives.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fmt;

use quantinfer_core::{
    DType, Q4KBlock, Q4_0Block, Q8_0Block, QuantizationScheme, QuantizedTensor, QuantizedTensor4,
    QuantizedTensorQ4K, Tensor, TensorStorageDescriptor, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE,
    Q8_0_BLOCK_SIZE,
};

pub type Result<T> = std::result::Result<T, QuantizationError>;

#[derive(Debug, Clone, PartialEq)]
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
    sum_abs: Vec<f32>,
    sum_square: Vec<f32>,
}

impl ImatrixStatistics {
    pub fn from_dataset(dataset: &CalibrationDataset) -> Result<Self> {
        let elements = dataset.samples()[0].shape().num_elements();
        let mut statistics = Self {
            count: 0,
            sum_abs: vec![0.0; elements],
            sum_square: vec![0.0; elements],
        };
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
        for (index, value) in values.iter().copied().enumerate() {
            self.sum_abs[index] += value.abs();
            self.sum_square[index] += value * value;
        }
        self.count += 1;
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn mean_abs(&self) -> Vec<f32> {
        let denominator = self.count.max(1) as f32;
        self.sum_abs
            .iter()
            .map(|value| value / denominator)
            .collect()
    }

    pub fn mean_square(&self) -> Vec<f32> {
        let denominator = self.count.max(1) as f32;
        self.sum_square
            .iter()
            .map(|value| value / denominator)
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

pub fn quantize_q8_0(input: &Tensor) -> Result<QuantizedTensor> {
    let values = input.as_f32()?;
    if !values.len().is_multiple_of(Q8_0_BLOCK_SIZE) {
        return Err(QuantizationError::InvalidInput(
            "Q8_0 requires a multiple of 32 elements".to_owned(),
        ));
    }
    let mut blocks = Vec::with_capacity(values.len() / Q8_0_BLOCK_SIZE);
    for chunk in values.as_chunks::<Q8_0_BLOCK_SIZE>().0 {
        let maximum = chunk.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 127.0 };
        let mut quantized = [0_i8; Q8_0_BLOCK_SIZE];
        for (index, value) in chunk.iter().copied().enumerate() {
            let rounded = (value / scale).round().clamp(-127.0, 127.0);
            quantized[index] = rounded as i8;
        }
        blocks.push(Q8_0Block {
            scale,
            values: quantized,
        });
    }
    Ok(QuantizedTensor::from_blocks(input.shape().clone(), blocks)?)
}

pub fn dequantize_q8_0(input: &QuantizedTensor) -> Result<Tensor> {
    let values = input
        .blocks()
        .iter()
        .flat_map(|block| {
            block
                .values
                .iter()
                .map(move |value| block.scale * f32::from(*value))
        })
        .collect();
    Ok(Tensor::from_f32(input.shape().clone(), values)?)
}

pub fn quantize_q4_0(input: &Tensor) -> Result<QuantizedTensor4> {
    let values = input.as_f32()?;
    if !values.len().is_multiple_of(Q4_0_BLOCK_SIZE) {
        return Err(QuantizationError::InvalidInput(
            "Q4_0 requires a multiple of 32 elements".to_owned(),
        ));
    }
    let mut blocks = Vec::with_capacity(values.len() / Q4_0_BLOCK_SIZE);
    for chunk in values.as_chunks::<Q4_0_BLOCK_SIZE>().0 {
        let maximum = chunk.iter().copied().map(f32::abs).fold(0.0, f32::max);
        let scale = if maximum == 0.0 { 1.0 } else { maximum / 7.0 };
        let mut quantized = [0_u8; Q4_0_BLOCK_SIZE];
        for (index, value) in chunk.iter().copied().enumerate() {
            quantized[index] = ((value / scale).round().clamp(-8.0, 7.0) as i32 + 8) as u8;
        }
        blocks.push(Q4_0Block {
            scale,
            values: quantized,
        });
    }
    Ok(QuantizedTensor4::from_blocks(
        input.shape().clone(),
        blocks,
    )?)
}

pub fn dequantize_q4_0(input: &QuantizedTensor4) -> Result<Tensor> {
    let values = input
        .blocks()
        .iter()
        .flat_map(|block| {
            block
                .values
                .iter()
                .map(move |value| block.scale * (f32::from(*value) - 8.0))
        })
        .collect();
    Ok(Tensor::from_f32(input.shape().clone(), values)?)
}

pub fn quantize_q4_k(input: &Tensor) -> Result<QuantizedTensorQ4K> {
    let values = input.as_f32()?;
    if !values.len().is_multiple_of(Q4_K_BLOCK_SIZE) {
        return Err(QuantizationError::InvalidInput(
            "Q4_K requires a multiple of 256 elements".to_owned(),
        ));
    }
    let mut blocks = Vec::with_capacity(values.len() / Q4_K_BLOCK_SIZE);
    for chunk in values.as_chunks::<Q4_K_BLOCK_SIZE>().0 {
        let mut scales = [0.0_f32; 8];
        let mut mins = [0.0_f32; 8];
        let mut quantized = [0_u8; Q4_K_BLOCK_SIZE];
        for (subblock, values) in chunk.as_chunks::<32>().0.iter().enumerate() {
            let min = values
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min)
                .min(0.0);
            let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let scale = ((max - min) / 15.0).max(f32::EPSILON);
            scales[subblock] = scale;
            mins[subblock] = -min;
            for (index, value) in values.iter().copied().enumerate() {
                quantized[subblock * 32 + index] =
                    ((value - min) / scale).round().clamp(0.0, 15.0) as u8;
            }
        }
        let scale_max = scales.iter().copied().fold(0.0, f32::max).max(f32::EPSILON);
        let min_max = mins.iter().copied().fold(0.0, f32::max);
        let d = scale_max / 63.0;
        let dmin = if min_max == 0.0 { 1.0 } else { min_max / 63.0 };
        let scale_values = scales.map(|value| (value / d).round().clamp(0.0, 63.0) as u8);
        let min_values = mins.map(|value| (value / dmin).round().clamp(0.0, 63.0) as u8);
        let mut packed_scales = [0_u8; 12];
        packed_scales[..4].copy_from_slice(&scale_values[..4]);
        packed_scales[4..8].copy_from_slice(&min_values[..4]);
        for index in 0..4 {
            packed_scales[8 + index] =
                (scale_values[4 + index] & 0x0f) | ((min_values[4 + index] & 0x0f) << 4);
            packed_scales[index] |= (scale_values[4 + index] >> 4) << 6;
            packed_scales[4 + index] |= (min_values[4 + index] >> 4) << 6;
        }
        let mut packed_values = [0_u8; Q4_K_BLOCK_SIZE / 2];
        for group in 0..4 {
            for index in 0..32 {
                let offset = group * 32 + index;
                packed_values[offset] =
                    quantized[group * 64 + index] | (quantized[group * 64 + 32 + index] << 4);
            }
        }
        blocks.push(Q4KBlock {
            scale: d,
            min_scale: dmin,
            scales: packed_scales,
            values: packed_values,
        });
    }
    Ok(QuantizedTensorQ4K::from_blocks(
        input.shape().clone(),
        blocks,
    )?)
}

pub fn dequantize_q4_k(input: &QuantizedTensorQ4K) -> Result<Tensor> {
    let mut output = Vec::with_capacity(input.num_elements());
    for block in input.blocks() {
        let mut scales = [0_u8; 8];
        let mut mins = [0_u8; 8];
        for index in 0..4 {
            scales[index] = block.scales[index] & 0x3f;
            mins[index] = block.scales[4 + index] & 0x3f;
        }
        for index in 0..4 {
            scales[4 + index] =
                (block.scales[8 + index] & 0x0f) | ((block.scales[index] >> 6) << 4);
            mins[4 + index] =
                (block.scales[8 + index] >> 4) | ((block.scales[4 + index] >> 6) << 4);
        }
        for index in 0..Q4_K_BLOCK_SIZE {
            let group = index / 64;
            let within_group = index % 64;
            let packed = block.values[group * 32 + within_group % 32];
            let quantized = if within_group < 32 {
                packed & 0x0f
            } else {
                packed >> 4
            };
            let subblock = index / 32;
            let scale = block.scale * f32::from(scales[subblock]);
            let minimum = block.min_scale * f32::from(mins[subblock]);
            output.push(scale * f32::from(quantized) - minimum);
        }
    }
    Ok(Tensor::from_f32(input.shape().clone(), output)?)
}

#[derive(Debug, Clone, PartialEq)]
pub enum QuantizedData {
    Float(Tensor),
    Q8_0(QuantizedTensor),
    Q4_0(QuantizedTensor4),
    #[allow(non_camel_case_types)]
    Q4_K(QuantizedTensorQ4K),
}

impl QuantizedData {
    pub fn scheme(&self) -> QuantizationScheme {
        match self {
            Self::Float(_) => QuantizationScheme::None,
            Self::Q8_0(_) => QuantizationScheme::Q8_0,
            Self::Q4_0(_) => QuantizationScheme::Q4_0,
            Self::Q4_K(_) => QuantizationScheme::Q4_K,
        }
    }

    pub fn shape(&self) -> &quantinfer_core::Shape {
        match self {
            Self::Float(tensor) => tensor.shape(),
            Self::Q8_0(tensor) => tensor.shape(),
            Self::Q4_0(tensor) => tensor.shape(),
            Self::Q4_K(tensor) => tensor.shape(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationStats {
    pub elements: usize,
    pub min: f32,
    pub max: f32,
    pub mean_abs: f32,
    pub max_abs: f32,
}

impl QuantizationStats {
    pub fn inspect(input: &Tensor) -> Result<Self> {
        let values = input.as_f32()?;
        if values.is_empty() {
            return Err(QuantizationError::InvalidInput(
                "cannot inspect an empty tensor".to_owned(),
            ));
        }
        let min = values.iter().copied().fold(f32::INFINITY, f32::min);
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum_abs = values.iter().map(|value| value.abs()).sum::<f32>();
        Ok(Self {
            elements: values.len(),
            min,
            max,
            mean_abs: sum_abs / values.len() as f32,
            max_abs: min.abs().max(max.abs()),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationErrorReport {
    pub max_absolute_error: f32,
    pub mean_absolute_error: f32,
    pub relative_l2_error: f32,
}

pub trait QuantizationProvider {
    fn scheme(&self) -> QuantizationScheme;
    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor>;
    fn inspect(&self, input: &Tensor) -> Result<QuantizationStats> {
        QuantizationStats::inspect(input)
    }
    fn validate(&self, input: &Tensor) -> Result<()>;
    fn quantize(&self, input: &Tensor) -> Result<QuantizedData>;
    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor>;
    fn error_report(
        &self,
        reference: &Tensor,
        input: &QuantizedData,
    ) -> Result<QuantizationErrorReport> {
        let restored = self.dequantize(input)?;
        let reference_values = reference.as_f32()?;
        let restored_values = restored.as_f32()?;
        if reference_values.len() != restored_values.len() {
            return Err(QuantizationError::InvalidInput(
                "error report tensors must have the same number of elements".to_owned(),
            ));
        }
        let mut max_absolute_error: f32 = 0.0;
        let mut sum_absolute_error: f32 = 0.0;
        let mut sum_squared_reference: f32 = 0.0;
        let mut sum_squared_error: f32 = 0.0;
        for (reference, restored) in reference_values.iter().zip(restored_values) {
            let error = *reference - *restored;
            max_absolute_error = max_absolute_error.max(error.abs());
            sum_absolute_error += error.abs();
            sum_squared_reference += reference * reference;
            sum_squared_error += error * error;
        }
        let denominator = sum_squared_reference.sqrt().max(f32::EPSILON);
        Ok(QuantizationErrorReport {
            max_absolute_error,
            mean_absolute_error: sum_absolute_error / reference_values.len().max(1) as f32,
            relative_l2_error: sum_squared_error.sqrt() / denominator,
        })
    }
    fn export_layout(&self) -> &'static str;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FloatProvider;

impl QuantizationProvider for FloatProvider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::None
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::floating(DType::F32, shape)?)
    }

    fn validate(&self, _input: &Tensor) -> Result<()> {
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Float(input.clone()))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Float(tensor) => Ok(tensor.clone()),
            _ => Err(QuantizationError::InvalidInput(
                "FloatProvider cannot dequantize a quantized tensor".to_owned(),
            )),
        }
    }

    fn export_layout(&self) -> &'static str {
        "f32"
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeQ8_0Provider;

impl QuantizationProvider for NativeQ8_0Provider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q8_0
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::quantized(
            DType::Q8_0,
            QuantizationScheme::Q8_0,
            shape,
            Q8_0_BLOCK_SIZE,
        )?)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        if !input.as_f32()?.len().is_multiple_of(Q8_0_BLOCK_SIZE) {
            return Err(QuantizationError::InvalidInput(
                "Q8_0 requires a multiple of 32 elements".to_owned(),
            ));
        }
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Q8_0(quantize_q8_0(input)?))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Q8_0(tensor) => dequantize_q8_0(tensor),
            _ => Err(QuantizationError::InvalidInput(
                "NativeQ8_0Provider expects Q8_0 data".to_owned(),
            )),
        }
    }

    fn export_layout(&self) -> &'static str {
        "ggml_q8_0"
    }
}

/// The optional GGML adapter currently shares the validated native Rust
/// implementation. A future FFI feature can replace its packing routine
/// without changing the provider contract or runtime representation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GgmlQ8_0Provider;

impl QuantizationProvider for GgmlQ8_0Provider {
    fn scheme(&self) -> QuantizationScheme {
        NativeQ8_0Provider.scheme()
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        NativeQ8_0Provider.storage_descriptor(shape)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        NativeQ8_0Provider.validate(input)
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        NativeQ8_0Provider.quantize(input)
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        NativeQ8_0Provider.dequantize(input)
    }

    fn export_layout(&self) -> &'static str {
        "ggml_q8_0"
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeQ4_0Provider;

impl QuantizationProvider for NativeQ4_0Provider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q4_0
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::quantized(
            DType::Q4_0,
            QuantizationScheme::Q4_0,
            shape,
            Q4_0_BLOCK_SIZE,
        )?)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        if !input.as_f32()?.len().is_multiple_of(Q4_0_BLOCK_SIZE) {
            return Err(QuantizationError::InvalidInput(
                "Q4_0 requires a multiple of 32 elements".to_owned(),
            ));
        }
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Q4_0(quantize_q4_0(input)?))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Q4_0(tensor) => dequantize_q4_0(tensor),
            _ => Err(QuantizationError::InvalidInput(
                "NativeQ4_0Provider expects Q4_0 data".to_owned(),
            )),
        }
    }

    fn export_layout(&self) -> &'static str {
        "ggml_q4_0"
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NativeQ4KProvider;

impl QuantizationProvider for NativeQ4KProvider {
    fn scheme(&self) -> QuantizationScheme {
        QuantizationScheme::Q4_K
    }

    fn storage_descriptor(&self, shape: quantinfer_core::Shape) -> Result<TensorStorageDescriptor> {
        Ok(TensorStorageDescriptor::quantized(
            DType::Q4_K,
            QuantizationScheme::Q4_K,
            shape,
            Q4_K_BLOCK_SIZE,
        )?)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        if !input.as_f32()?.len().is_multiple_of(Q4_K_BLOCK_SIZE) {
            return Err(QuantizationError::InvalidInput(
                "Q4_K requires a multiple of 256 elements".to_owned(),
            ));
        }
        Ok(())
    }

    fn quantize(&self, input: &Tensor) -> Result<QuantizedData> {
        self.validate(input)?;
        Ok(QuantizedData::Q4_K(quantize_q4_k(input)?))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        match input {
            QuantizedData::Q4_K(tensor) => dequantize_q4_k(tensor),
            _ => Err(QuantizationError::InvalidInput(
                "NativeQ4KProvider expects Q4_K data".to_owned(),
            )),
        }
    }

    fn export_layout(&self) -> &'static str {
        "ggml_q4_k"
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dequantize_q8_0, quantize_q8_0, CalibrationDataset, ImatrixStatistics, MixedPrecisionPlan,
        NativeQ4KProvider, NativeQ4_0Provider, QuantizationProvider,
    };
    use quantinfer_core::{QuantizationScheme, Tensor};

    #[test]
    fn quantizes_and_dequantizes_a_block() {
        let input =
            Tensor::from_f32(vec![32], (0..32).map(|value| value as f32).collect()).unwrap();
        let quantized = quantize_q8_0(&input).unwrap();
        assert_eq!(quantized.blocks().len(), 1);
        let restored = dequantize_q8_0(&quantized).unwrap();
        let values = restored.as_f32().unwrap();
        assert!((values[31] - 31.0).abs() < 0.3);
    }

    #[test]
    fn rejects_non_block_aligned_input() {
        let input = Tensor::from_f32(vec![31], vec![1.0; 31]).unwrap();
        assert!(quantize_q8_0(&input).is_err());
    }

    #[test]
    fn quantizes_q4_0_and_reports_error() {
        let input = Tensor::from_f32(
            vec![32],
            (0..32).map(|value| value as f32 / 4.0 - 4.0).collect(),
        )
        .unwrap();
        let provider = NativeQ4_0Provider;
        let quantized = provider.quantize(&input).unwrap();
        let restored = provider.dequantize(&quantized).unwrap();
        assert!(
            provider
                .error_report(&input, &quantized)
                .unwrap()
                .max_absolute_error
                < 0.4
        );
        assert_eq!(restored.shape(), input.shape());
        assert_eq!(provider.export_layout(), "ggml_q4_0");
    }

    #[test]
    fn builds_calibration_statistics_and_mixed_precision_assignments() {
        let dataset = CalibrationDataset::new(vec![
            Tensor::from_f32(vec![2], vec![-1.0, 2.0]).unwrap(),
            Tensor::from_f32(vec![2], vec![3.0, -4.0]).unwrap(),
        ])
        .unwrap();
        let statistics = ImatrixStatistics::from_dataset(&dataset).unwrap();
        assert_eq!(statistics.count(), 2);
        assert_eq!(statistics.mean_abs(), vec![2.0, 3.0]);
        let mut plan = MixedPrecisionPlan::default();
        plan.assign("blk.0.attn_q.weight", QuantizationScheme::Q8_0);
        assert_eq!(
            plan.scheme_for("blk.0.attn_q.weight"),
            Some(QuantizationScheme::Q8_0)
        );
    }

    #[test]
    fn quantizes_and_dequantizes_standard_q4_k_blocks() {
        let input = Tensor::from_f32(
            vec![256],
            (0..256).map(|index| index as f32 / 128.0 - 1.0).collect(),
        )
        .unwrap();
        let provider = NativeQ4KProvider;
        let quantized = provider.quantize(&input).unwrap();
        let report = provider.error_report(&input, &quantized).unwrap();
        assert!(report.max_absolute_error < 0.2);
        assert_eq!(provider.export_layout(), "ggml_q4_k");
    }
}
