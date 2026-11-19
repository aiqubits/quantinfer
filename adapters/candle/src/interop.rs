mod qif;
pub use qif::{packed_from_qif, packed_view_from_qif, PackedQifTensor};

use std::fmt;

use candle_core::{DType as CandleDType, Device, Tensor as CandleTensor};
use quantinfer_core::{CoreError, Tensor};

pub type Result<T> = std::result::Result<T, InteropError>;

#[derive(Debug)]
pub enum InteropError {
    Candle(candle_core::Error),
    Core(CoreError),
    Format(quantinfer_format::FormatError),
    Invalid(String),
}

impl fmt::Display for InteropError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Candle(error) => error.fmt(formatter),
            Self::Core(error) => error.fmt(formatter),
            Self::Format(error) => error.fmt(formatter),
            Self::Invalid(message) => formatter.write_str(message),
        }
    }
}

impl std::error::Error for InteropError {}

/// Copy a QuantInfer F32 tensor to the requested Candle device.
pub fn to_candle(tensor: &Tensor, device: &Device) -> Result<CandleTensor> {
    let values = tensor.as_f32().map_err(InteropError::Core)?;
    CandleTensor::from_slice(values, tensor.shape().dims(), device).map_err(InteropError::Candle)
}

/// Copy a floating-point Candle tensor to QuantInfer F32 in logical shape order.
pub fn from_candle(tensor: &CandleTensor) -> Result<Tensor> {
    if !matches!(
        tensor.dtype(),
        CandleDType::F32 | CandleDType::F16 | CandleDType::BF16
    ) {
        return Err(InteropError::Invalid(
            "Candle tensor must use F32, F16 or BF16".to_owned(),
        ));
    }
    let values = tensor
        .to_device(&Device::Cpu)
        .and_then(|cpu| cpu.to_dtype(CandleDType::F32))
        .and_then(|float| float.flatten_all())
        .and_then(|flat| flat.to_vec1::<f32>())
        .map_err(InteropError::Candle)?;
    Tensor::from_f32(tensor.dims().to_vec(), values).map_err(InteropError::Core)
}

#[cfg(test)]
mod tests;
