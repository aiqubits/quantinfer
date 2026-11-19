use std::collections::BTreeMap;

use quantinfer_core::{QuantizationScheme, Shape, TensorStorageDescriptor};
use quantinfer_format::{qif::write_qif, ModelManifest, TensorManifest};

use super::*;
use candle_core::quantized::GgmlDType;
use quantinfer_core::DType;
use quantinfer_format::qif::Reader as QifReader;

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
    manifest.quantization = QuantizationScheme::Q4_0;
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
