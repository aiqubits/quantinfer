use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{
    DType, QuantizationScheme, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
use quantinfer_format::{GgmlType, GgufReader, ModelManifest};
use quantinfer_runtime::{ExecutionRequest, ExecutionResolver, Operation, Result, RuntimeError};

/// Select a CPU kernel that supports the weight types recorded in a GGUF.
pub fn resolve_backend(reader: &GgufReader, manifest: Option<&ModelManifest>) -> Result<CpuScalar> {
    let resolver = ExecutionResolver::with_default_cpu();
    let mut request = ExecutionRequest::cpu(DType::F32, QuantizationScheme::None);
    request.tensor_layout = "row_major".to_owned();
    request.operation = Some(Operation::Matmul);
    if let Some(manifest) = manifest {
        request.head_dimension = manifest.kernel_requirements.head_dimension;
        request.maximum_sequence_length = manifest.kernel_requirements.maximum_sequence_length;
        request.alignment = manifest.kernel_requirements.alignment;
        if manifest
            .tensors
            .iter()
            .any(|tensor| tensor.descriptor.physical_layout != request.tensor_layout)
        {
            return Err(RuntimeError::InvalidInput(
                "model package declares an unsupported tensor layout".to_owned(),
            ));
        }
    }
    let plan = resolver.resolve(&request)?;

    let mut checked = Vec::new();
    for tensor in reader.tensors() {
        if checked.contains(&tensor.dtype) {
            continue;
        }
        checked.push(tensor.dtype);
        let (dtype, scheme, block_size) = match tensor.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => continue,
            GgmlType::Q8_0 => (DType::Q8_0, QuantizationScheme::Q8_0, Q8_0_BLOCK_SIZE),
            GgmlType::Q4_0 => (DType::Q4_0, QuantizationScheme::Q4_0, Q4_0_BLOCK_SIZE),
            GgmlType::Q4_K => (DType::Q4_K, QuantizationScheme::Q4_K, Q4_K_BLOCK_SIZE),
            GgmlType::Q5_K => (DType::Q5_K, QuantizationScheme::Q5_K, Q4_K_BLOCK_SIZE),
            GgmlType::Q6_K => (DType::Q6_K, QuantizationScheme::Q6_K, Q4_K_BLOCK_SIZE),
        };
        request.dtype = dtype;
        request.quantization = scheme;
        request.operation = Some(Operation::QuantizedMatmul);
        request.alignment = Some(block_size);
        let quantized_plan = resolver.resolve(&request)?;
        if quantized_plan.provider_name != plan.provider_name {
            return Err(RuntimeError::InvalidInput(
                "model tensors require incompatible CPU providers".to_owned(),
            ));
        }
    }

    if let Some(manifest) = manifest {
        if !manifest.execution_variants.is_empty()
            && !manifest.execution_variants.iter().any(|variant| {
                variant.runtime_provider == plan.provider_name
                    && variant.kernel_provider == "cpu-scalar"
                    && variant.quantization == manifest.quantization
            })
        {
            return Err(RuntimeError::InvalidInput(format!(
                "model package {} has no compatible CPU execution variant",
                manifest.model_id
            )));
        }
    }
    match plan.provider_name.as_str() {
        "cpu-scalar" => Ok(CpuScalar::new()),
        provider => Err(RuntimeError::InvalidInput(format!(
            "CPU kernel factory is unavailable for provider {provider}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use quantinfer_format::{ExecutionVariant, GgufReader, GgufWriter, ModelManifest};

    use super::*;

    #[test]
    fn selects_cpu_for_legacy_packages_and_checks_declared_variants() {
        let reader = GgufReader::open(GgufWriter::new().finish().unwrap()).unwrap();
        let mut manifest = ModelManifest::new("fixture", "qwen3");
        assert!(resolve_backend(&reader, Some(&manifest)).is_ok());
        manifest.execution_variants.push(ExecutionVariant {
            name: "accelerator".to_owned(),
            quantization: QuantizationScheme::None,
            dtype: "f32-accumulation".to_owned(),
            runtime_provider: "cuda".to_owned(),
            kernel_provider: "cuda".to_owned(),
        });
        assert!(resolve_backend(&reader, Some(&manifest)).is_err());
        assert!(resolve_backend(&reader, None).is_ok());
    }
}
