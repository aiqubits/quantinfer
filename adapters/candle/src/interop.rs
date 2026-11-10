use std::borrow::Cow;
use std::fmt;

use candle_core::quantized::{GgmlDType, QStorage, QTensor};
use candle_core::{DType as CandleDType, Device, Tensor as CandleTensor};
use quantinfer_core::{CoreError, DType, Tensor};
use quantinfer_format::qif::QifReader;

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

/// Copy a verified QIF tensor into Candle packed storage without requantizing.
pub fn packed_from_qif(reader: &QifReader<'_>, name: &str, device: &Device) -> Result<QTensor> {
    if cfg!(target_endian = "big") {
        return Err(InteropError::Invalid(
            "packed Candle interchange requires a little-endian target".to_owned(),
        ));
    }
    let bytes = reader.tensor_bytes(name).map_err(InteropError::Format)?;
    let descriptor = &reader
        .manifest()
        .tensors
        .iter()
        .find(|tensor| tensor.name == name)
        .ok_or_else(|| InteropError::Invalid(format!("unknown QIF tensor {name}")))?
        .descriptor;
    let dtype = match descriptor.dtype {
        DType::Q8_0 => GgmlDType::Q8_0,
        DType::Q4_0 => GgmlDType::Q4_0,
        DType::Q4_K => GgmlDType::Q4K,
        DType::Q5_K => GgmlDType::Q5K,
        DType::Q6_K => GgmlDType::Q6K,
        _ => {
            return Err(InteropError::Invalid(
                "QIF tensor does not use a supported packed quantization".to_owned(),
            ));
        }
    };
    let shape = descriptor.logical_shape.dims();
    if shape
        .last()
        .is_none_or(|width| !width.is_multiple_of(dtype.block_size()))
    {
        return Err(InteropError::Invalid(
            "Candle requires quantization blocks to fit within each tensor row".to_owned(),
        ));
    }
    let expected = descriptor.logical_shape.num_elements() / dtype.block_size() * dtype.type_size();
    if bytes.len() != expected {
        return Err(InteropError::Invalid(
            "QIF and Candle block sizes disagree".to_owned(),
        ));
    }

    // Candle expects native alignment. A caller may open QIF from a sub-slice.
    let aligned;
    let bytes = if bytes.as_ptr().cast::<u16>().is_aligned() {
        bytes
    } else {
        aligned = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| u16::from_ne_bytes(*pair))
            .collect::<Vec<_>>();
        bytemuck::cast_slice(&aligned)
    };
    QStorage::from_data(Cow::Borrowed(bytes), device, dtype)
        .and_then(|storage| QTensor::new(storage, shape))
        .map_err(InteropError::Candle)
}

const _: () = {
    use candle_core::quantized::k_quants::{BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8_0};
    assert!(align_of::<BlockQ4_0>() == align_of::<u16>());
    assert!(align_of::<BlockQ8_0>() == align_of::<u16>());
    assert!(align_of::<BlockQ4K>() == align_of::<u16>());
    assert!(align_of::<BlockQ5K>() == align_of::<u16>());
    assert!(align_of::<BlockQ6K>() == align_of::<u16>());
};

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use quantinfer_core::{QuantizationScheme, Shape, TensorStorageDescriptor};
    use quantinfer_format::{qif::write_qif, ModelManifest, TensorManifest};

    use super::*;

    #[test]
    fn round_trips_logical_values_of_a_transposed_tensor() {
        let candle = CandleTensor::from_vec(vec![1.0f32, 2., 3., 4., 5., 6.], (2, 3), &Device::Cpu)
            .unwrap()
            .transpose(0, 1)
            .unwrap()
            .to_dtype(CandleDType::F16)
            .unwrap();
        let native = from_candle(&candle).unwrap();
        assert_eq!(native.shape().dims(), [3, 2]);
        assert_eq!(native.as_f32().unwrap(), [1., 4., 2., 5., 3., 6.]);
        let exported = to_candle(&native, &Device::Cpu).unwrap();
        assert_eq!(
            exported.to_vec2::<f32>().unwrap(),
            vec![vec![1., 4.], vec![2., 5.], vec![3., 6.]]
        );
    }

    #[test]
    #[cfg(target_endian = "little")]
    fn imports_unaligned_qif_blocks_without_requantizing() {
        let mut manifest = ModelManifest::new("fixture", "llama");
        manifest.tensors.push(TensorManifest {
            name: "weight".to_owned(),
            descriptor: TensorStorageDescriptor::quantized(
                DType::Q4_0,
                QuantizationScheme::Q4_0,
                Shape::from(vec![1, 32]),
                32,
            )
            .unwrap(),
            checksum_sha256: None,
        });
        let mut block = [0xb9; 18];
        block[..2].copy_from_slice(&[0, 0x3c]);
        let mut bytes = Vec::<u8>::with_capacity(2048);
        let offset = usize::from(bytes.as_ptr().cast::<u16>().is_aligned());
        bytes.resize(offset, 0);
        write_qif(manifest, &BTreeMap::new(), |_| Ok(&block), &mut bytes).unwrap();
        let qif = QifReader::open(&bytes[offset..]).unwrap();
        assert!(!qif
            .tensor_bytes("weight")
            .unwrap()
            .as_ptr()
            .cast::<u16>()
            .is_aligned());
        let packed = packed_from_qif(&qif, "weight", &Device::Cpu).unwrap();
        assert_eq!(packed.dtype(), GgmlDType::Q4_0);
        assert_eq!(packed.data().unwrap().as_ref(), block);
        let decoded = from_candle(&packed.dequantize(&Device::Cpu).unwrap()).unwrap();
        assert_eq!(decoded.as_f32().unwrap(), [[1.0; 16], [3.0; 16]].concat());
    }
}
