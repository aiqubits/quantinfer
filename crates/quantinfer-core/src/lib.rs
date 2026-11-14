//! Core tensor, layout, device, and quantized storage primitives.

#![forbid(unsafe_code)]

mod error;
mod quantized;
mod tensor;

pub use error::{CoreError, Result};
pub use quantized::{
    Q4KBlock, Q4_0Block, Q5KBlock, Q6KBlock, Q8_0Block, QuantizationAxis, QuantizationScheme,
    QuantizedTensor, QuantizedTensor4, QuantizedTensorDescriptor, QuantizedTensorQ4K,
    QuantizedTensorQ5K, QuantizedTensorQ6K, ScaleType, StorageType, TensorStorageDescriptor,
    ZeroPointType, GGML_OUTPUT_ROWS_LAYOUT, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
pub use tensor::{DType, Device, Shape, Storage, Strides, Tensor, TensorView};

#[cfg(test)]
mod tests {
    use super::{DType, Shape, Tensor};

    #[test]
    fn creates_and_indexes_contiguous_tensor() {
        let tensor = Tensor::from_f32(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(tensor.dtype(), DType::F32);
        assert_eq!(tensor.get_f32(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn reshape_preserves_values() {
        let tensor = Tensor::from_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let reshaped = tensor.reshape(Shape::new(vec![4]).unwrap()).unwrap();
        assert_eq!(reshaped.as_f32().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn validates_quantized_storage_descriptor() {
        let descriptor = super::TensorStorageDescriptor::quantized(
            DType::Q8_0,
            super::QuantizationScheme::Q8_0,
            Shape::from(vec![64]),
            32,
        )
        .unwrap();
        descriptor.validate().unwrap();
        assert_eq!(descriptor.dtype.bit_width(), 8);
        assert!(super::TensorStorageDescriptor::quantized(
            DType::Q4_0,
            super::QuantizationScheme::Q4_0,
            Shape::from(vec![31]),
            32,
        )
        .is_err());
        let mut invalid = descriptor;
        invalid.quantization = super::QuantizationScheme::Q4_0;
        assert!(invalid.validate().is_err());
        for (dtype, scheme) in [
            (DType::Q5_K, super::QuantizationScheme::Q5_K),
            (DType::Q6_K, super::QuantizationScheme::Q6_K),
        ] {
            let descriptor = super::TensorStorageDescriptor::quantized(
                dtype,
                scheme,
                Shape::from(vec![2, 256]),
                super::Q4_K_BLOCK_SIZE,
            )
            .unwrap();
            assert_eq!(descriptor.quantization, scheme);
        }
    }

    #[test]
    fn rejects_overflowing_shapes() {
        assert!(Shape::new(vec![usize::MAX, 2]).is_err());
        assert!(Tensor::from_f32(Shape::from(vec![usize::MAX, 2]), Vec::new()).is_err());
    }
}
