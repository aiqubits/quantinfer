use std::mem::size_of;

use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{
    DType, Q4KBlock, Q4_0Block, Q5KBlock, Q6KBlock, Q8_0Block, QuantizationScheme, Q4_0_BLOCK_SIZE,
    Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
use quantinfer_format::{GgmlType, GgufReader, GgufValue, ModelManifest};
use quantinfer_model::{ModelSpec, TensorRole};
use quantinfer_runtime::llama::uses_ggml_layout;
use quantinfer_runtime::{ExecutionRequest, ExecutionResolver, Operation, Result, RuntimeError};

/// Select a CPU kernel that supports the weight types recorded in a GGUF.
pub fn resolve_backend(reader: &GgufReader, manifest: Option<&ModelManifest>) -> Result<CpuScalar> {
    let resolver = ExecutionResolver::with_default_cpu();
    let mut request = ExecutionRequest::cpu(DType::F32, QuantizationScheme::None);
    let is_llama = manifest.is_some_and(|manifest| manifest.architecture == "llama")
        || matches!(reader.metadata().get("general.architecture"), Some(GgufValue::String(value)) if value == "llama");
    let ggml_layout = is_llama && uses_ggml_layout(reader)?;
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
        if ggml_layout && ggml_float_weight(&tensor.name) {
            continue;
        }
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
        if ggml_layout {
            request.tensor_layout = "ggml_output_rows".to_owned();
        }
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

fn ggml_float_weight(name: &str) -> bool {
    matches!(name, "token_embd.weight" | "output.weight")
}

/// Estimate the storage used by the CPU representation after GGUF decoding.
pub fn estimated_weight_bytes(reader: &GgufReader, spec: &ModelSpec) -> Result<usize> {
    let invalid = || RuntimeError::InvalidInput("CPU weight size overflows usize".to_owned());
    let mut bytes = 0usize;
    let ggml_layout = spec.architecture == "llama" && uses_ggml_layout(reader)?;
    for tensor in reader.tensors() {
        let elements: usize = tensor.shape.iter().product();
        let (block_size, native_bytes) = match tensor.dtype {
            GgmlType::F32 | GgmlType::F16 | GgmlType::BF16 => (1, size_of::<f32>()),
            GgmlType::Q8_0 => (Q8_0_BLOCK_SIZE, size_of::<Q8_0Block>()),
            GgmlType::Q4_0 => (Q4_0_BLOCK_SIZE, size_of::<Q4_0Block>()),
            GgmlType::Q4_K => (Q4_K_BLOCK_SIZE, size_of::<Q4KBlock>()),
            GgmlType::Q5_K => (Q4_K_BLOCK_SIZE, size_of::<Q5KBlock>()),
            GgmlType::Q6_K => (Q4_K_BLOCK_SIZE, size_of::<Q6KBlock>()),
        };
        let tensor_bytes = if ggml_layout && ggml_float_weight(&tensor.name) {
            elements.checked_mul(size_of::<f32>())
        } else {
            (elements / block_size).checked_mul(native_bytes)
        }
        .ok_or_else(invalid)?;
        bytes = bytes.checked_add(tensor_bytes).ok_or_else(invalid)?;
    }
    if spec.embedding.tied_lm_head && reader.tensor_info("output.weight").is_err() {
        let embedding = spec
            .tensor(TensorRole::TokenEmbedding, None)
            .ok_or_else(|| RuntimeError::InvalidInput("model has no embedding".to_owned()))?;
        let head_bytes = embedding
            .logical_shape
            .dims()
            .iter()
            .try_fold(size_of::<f32>(), |bytes, dimension| {
                bytes.checked_mul(*dimension)
            })
            .ok_or_else(invalid)?;
        bytes = bytes.checked_add(head_bytes).ok_or_else(invalid)?;
    }
    Ok(bytes)
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
