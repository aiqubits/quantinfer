//! SignRound projection calibration for signed F16-scale QIF blocks.

mod optimize;
mod statistics;

pub use statistics::AutoRoundCalibration;

use quantinfer_core::{
    DType, QuantizationScheme, Shape, Tensor, TensorStorageDescriptor, OUTPUT_ROWS_LAYOUT,
};
use quantinfer_quant::{
    CalibratedQuantizationProvider, QuantizationError, QuantizationProvider, QuantizedData, Result,
};

#[derive(Debug, Clone, Copy)]
pub struct AutoRoundOptions {
    pub bits: usize,
    pub group_size: usize,
    pub iterations: usize,
    /// Defaults to 1 / iterations. Both step sizes decay linearly.
    pub rounding_lr: Option<f64>,
    pub clipping_lr: Option<f64>,
    pub threads: usize,
    pub maximum_workspace_bytes: usize,
}

impl Default for AutoRoundOptions {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 32,
            iterations: 200,
            rounding_lr: None,
            clipping_lr: None,
            threads: std::thread::available_parallelism().map_or(1, |n| n.get().min(8)),
            maximum_workspace_bytes: 512 * 1024 * 1024,
        }
    }
}

#[derive(Debug)]
pub struct AutoRoundResult {
    pub weights: QuantizedData,
    pub initial_mse: f64,
    pub best_mse: f64,
    /// Output rows with a strict improvement over their initial grid.
    pub improved_outputs: usize,
}

pub struct AutoRoundProvider {
    options: AutoRoundOptions,
    pool: rayon::ThreadPool,
}

impl AutoRoundProvider {
    pub fn new(options: AutoRoundOptions) -> Result<Self> {
        if !matches!(options.bits, 4 | 8)
            || options.group_size == 0
            || !options.group_size.is_multiple_of(32)
            || !(1..=4096).contains(&options.iterations)
            || !(1..=64).contains(&options.threads)
            || options.maximum_workspace_bytes == 0
            || [options.rounding_lr, options.clipping_lr]
                .into_iter()
                .flatten()
                .any(|lr| !lr.is_finite() || lr <= 0.0 || lr > 1.0)
        {
            return Err(invalid("invalid AutoRound bit width, grouping, iterations, threads, step size or workspace"));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(options.threads)
            .build()
            .map_err(|error| invalid(format!("cannot create AutoRound workers: {error}")))?;
        Ok(Self { options, pool })
    }

    pub fn options(&self) -> &AutoRoundOptions {
        &self.options
    }

    /// Optimize rounding offsets and symmetric clipping multipliers together.
    /// Use the complete linear output error on the supplied calibration rows.
    /// A row's selected state is evaluated on the same F16 grid it will store.
    pub fn optimize(
        &self,
        weight: &Tensor,
        calibration: &AutoRoundCalibration,
    ) -> Result<AutoRoundResult> {
        self.validate(weight)?;
        optimize::optimize(self, weight, calibration)
    }
}

impl QuantizationProvider for AutoRoundProvider {
    fn scheme(&self) -> QuantizationScheme {
        if self.options.bits == 4 {
            QuantizationScheme::Q4_0
        } else {
            QuantizationScheme::Q8_0
        }
    }

    fn storage_descriptor(&self, shape: Shape) -> Result<TensorStorageDescriptor> {
        matrix_shape(&shape)?;
        let dtype = if self.options.bits == 4 {
            DType::Q4_0
        } else {
            DType::Q8_0
        };
        let mut descriptor = TensorStorageDescriptor::quantized(dtype, self.scheme(), shape, 32)?;
        descriptor.physical_layout = OUTPUT_ROWS_LAYOUT.into();
        Ok(descriptor)
    }

    fn validate(&self, input: &Tensor) -> Result<()> {
        matrix_shape(input.shape())?;
        if input.as_f32()?.iter().any(|v| !v.is_finite()) {
            return Err(invalid("AutoRound weights must be finite F32"));
        }
        Ok(())
    }

    fn quantize(&self, _input: &Tensor) -> Result<QuantizedData> {
        Err(invalid(
            "AutoRound requires observed calibration activations",
        ))
    }

    fn dequantize(&self, input: &QuantizedData) -> Result<Tensor> {
        optimize::dequantize(input, self.options.bits)
    }

    fn export_layout(&self) -> &'static str {
        OUTPUT_ROWS_LAYOUT
    }
}

impl CalibratedQuantizationProvider for AutoRoundProvider {
    type Calibration = AutoRoundCalibration;

    fn quantize_calibrated(
        &self,
        input: &Tensor,
        calibration: &Self::Calibration,
    ) -> Result<QuantizedData> {
        Ok(self.optimize(input, calibration)?.weights)
    }
}

fn invalid(message: impl Into<String>) -> QuantizationError {
    QuantizationError::InvalidInput(message.into())
}

fn matrix_shape(shape: &Shape) -> Result<(usize, usize)> {
    match shape.dims() {
        &[k, n] if k > 0 && n > 0 && k.is_multiple_of(32) && k.checked_mul(n).is_some() => {
            Ok((k, n))
        }
        _ => Err(invalid(
            "AutoRound needs [input, output] with input width divisible by 32",
        )),
    }
}

fn buffer<T: Clone>(count: usize, value: T) -> Result<Vec<T>> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(count)
        .map_err(|_| invalid("cannot allocate AutoRound workspace"))?;
    result.resize(count, value);
    Ok(result)
}
