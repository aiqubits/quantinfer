//! GPU capability preflight, independent of device initialization.
use quantinfer_format::{GgufReader, ModelManifest};
use quantinfer_runtime::RuntimeProvider;

pub(super) fn validate<P: RuntimeProvider + ?Sized>(
    reader: &GgufReader,
    manifest: Option<&ModelManifest>,
    backend: &P,
) -> Result<(), String> {
    use quantinfer_core::{DType, Device, QuantizationScheme, OUTPUT_ROWS_LAYOUT};
    use quantinfer_format::GgmlType;
    use quantinfer_format::GGML_OUTPUT_ROWS_LAYOUT;
    use quantinfer_runtime::{ExecutionRequest, Operation};

    let manifest = manifest.or(reader.embedded_manifest());
    if let Some(manifest) = manifest {
        quantinfer_runtime::validate_manifest_for_runtime(manifest)
            .map_err(|error| error.to_string())?;
    }
    if let Some(manifest) = manifest {
        reader
            .validate_manifest(manifest)
            .map_err(|error| error.to_string())?;
    }
    let ggml = reader
        .uses_ggml_output_rows()
        .map_err(|error| error.to_string())?;
    let request = ExecutionRequest {
        device: Device::Gpu,
        tensor_layout: "row_major".to_owned(),
        operation: Some(Operation::Matmul),
        alignment: manifest.and_then(|manifest| manifest.kernel_requirements.alignment),
        head_dimension: manifest.and_then(|manifest| manifest.kernel_requirements.head_dimension),
        maximum_sequence_length: manifest
            .and_then(|manifest| manifest.kernel_requirements.maximum_sequence_length),
        ..ExecutionRequest::cpu(DType::F32, QuantizationScheme::None)
    };
    let plan = backend
        .resolve(&request)
        .map_err(|error| error.to_string())?;
    if let Some(manifest) = manifest {
        for operation in quantinfer_runtime::required_manifest_operations(manifest)
            .map_err(|error| error.to_string())?
        {
            if operation == Operation::QuantizedMatmul {
                if !reader.tensors().iter().any(|tensor| {
                    !matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
                }) {
                    return Err(
                        "required quantized_matmul has no quantized tensor dtype to resolve"
                            .to_owned(),
                    );
                }
                continue;
            }
            backend
                .resolve(&ExecutionRequest {
                    operation: Some(operation),
                    ..request.clone()
                })
                .map_err(|error| error.to_string())?;
        }
        if !manifest.execution_variants.is_empty()
            && !manifest.execution_variants.iter().any(|variant| {
                variant.runtime_provider == plan.provider_name
                    && variant.kernel_provider == plan.kernel_provider_name
                    && variant.quantization == manifest.quantization
            })
        {
            return Err(format!(
                "model package {} has no {} execution variant",
                manifest.model_id, plan.provider_name
            ));
        }
        if manifest.tensors.iter().any(|tensor| {
            let layout = tensor.descriptor.physical_layout.as_str();
            layout != "row_major" && !(ggml && layout == GGML_OUTPUT_ROWS_LAYOUT)
        }) {
            return Err(format!(
                "model package declares an unsupported {} tensor layout",
                plan.provider_name
            ));
        }
    }
    for (kind, dtype, quantization, block_size) in [
        (GgmlType::Q8_0, DType::Q8_0, QuantizationScheme::Q8_0, 32),
        (GgmlType::Q4_0, DType::Q4_0, QuantizationScheme::Q4_0, 32),
        (GgmlType::Q4_K, DType::Q4_K, QuantizationScheme::Q4_K, 256),
        (GgmlType::Q5_K, DType::Q5_K, QuantizationScheme::Q5_K, 256),
        (GgmlType::Q6_K, DType::Q6_K, QuantizationScheme::Q6_K, 256),
    ] {
        if reader.tensors().iter().any(|tensor| tensor.dtype == kind) {
            let tensor_layout = if ggml {
                OUTPUT_ROWS_LAYOUT
            } else {
                "row_major"
            };
            backend
                .resolve(&ExecutionRequest {
                    dtype,
                    quantization,
                    operation: Some(Operation::QuantizedMatmul),
                    tensor_layout: tensor_layout.to_owned(),
                    alignment: Some(block_size),
                    ..request
                })
                .map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use quantinfer::core::{
        DType, Device, QuantizationScheme, Shape, Tensor, TensorStorageDescriptor,
    };
    use quantinfer::quant::{quantize_q4_k, quantize_q5_k, quantize_q6_k};
    use quantinfer_format::{GgufWriter, TensorManifest};
    use quantinfer_runtime::{KernelCapability, Operation};
    use std::collections::BTreeSet;

    #[derive(Debug)]
    struct GpuFixture(Vec<KernelCapability>);

    impl RuntimeProvider for GpuFixture {
        fn name(&self) -> &str {
            "test-gpu"
        }
        fn capabilities(&self) -> &[KernelCapability] {
            &self.0
        }
    }

    fn provider() -> GpuFixture {
        GpuFixture(vec![KernelCapability {
            device: Device::Gpu,
            dtype: DType::F32,
            quantization: QuantizationScheme::None,
            tensor_layout: "row_major".into(),
            supported_operations: BTreeSet::from([Operation::Matmul]),
            alignment: 1,
            head_dimension: Some(64),
            maximum_sequence_length: Some(128),
        }])
    }

    #[test]
    fn unsupported_k_quant_weights_fail_preflight_without_opening_a_device() {
        let tensor = Tensor::from_f32(vec![16, 16], vec![0.01; 256]).unwrap();
        for kind in 0..3 {
            let mut writer = GgufWriter::new();
            match kind {
                0 => writer.add_q4_k_tensor("weight", &quantize_q4_k(&tensor).unwrap()),
                1 => writer.add_q5_k_tensor("weight", &quantize_q5_k(&tensor).unwrap()),
                _ => writer.add_q6_k_tensor("weight", &quantize_q6_k(&tensor).unwrap()),
            }
            let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
            assert!(validate(&reader, None, &provider()).is_err());
        }
    }

    #[test]
    fn gpu_plan_checks_declared_tensors_operations_and_kernel_limits() {
        let mut writer = GgufWriter::new();
        writer
            .add_f32_tensor("weight", vec![2], &[0.0, 1.0])
            .unwrap();
        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        let mut manifest = ModelManifest::new("fixture", "llama");
        manifest.tensors.push(TensorManifest {
            name: "weight".into(),
            descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2]))
                .unwrap(),
            checksum_sha256: None,
        });
        manifest
            .kernel_requirements
            .operations
            .push("matmul".into());
        validate(&reader, Some(&manifest), &provider()).unwrap();
        for field in 0..5 {
            let mut invalid = manifest.clone();
            match field {
                0 => invalid.tensors[0].descriptor.logical_shape = Shape::from(vec![3]),
                1 => {
                    invalid.tensors[0].descriptor =
                        TensorStorageDescriptor::floating(DType::F16, Shape::from(vec![2])).unwrap()
                }
                2 => invalid
                    .kernel_requirements
                    .operations
                    .push("attention".into()),
                3 => invalid.kernel_requirements.head_dimension = Some(128),
                _ => invalid.kernel_requirements.maximum_sequence_length = Some(256),
            }
            assert!(
                validate(&reader, Some(&invalid), &provider()).is_err(),
                "field {field}"
            );
        }
    }
}
