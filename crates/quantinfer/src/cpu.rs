use std::mem::size_of;

use quantinfer_backend_cpu::{CpuRuntimeConfig, CpuScalar};
use quantinfer_core::{
    DType, Q4KBlock, Q4_0Block, Q5KBlock, Q6KBlock, Q8_0Block, QuantizationScheme,
    GGML_OUTPUT_ROWS_LAYOUT, Q4_0_BLOCK_SIZE, Q4_K_BLOCK_SIZE, Q8_0_BLOCK_SIZE,
};
use quantinfer_format::{GgmlType, GgufReader, GgufValue, ModelManifest};
use quantinfer_model::{ModelSpec, TensorRole};
use quantinfer_runtime::llama::uses_ggml_layout;
use quantinfer_runtime::{
    ExecutionRequest, ExecutionResolver, InferenceProfile, Operation, Result, RuntimeError,
};

mod model;

pub use model::{load_gguf_tokenizer, model_architecture, CpuModel, DenseModel, ModelSource};

/// Select a CPU kernel that supports the weight types recorded in a GGUF.
pub fn resolve_backend(reader: &GgufReader, manifest: Option<&ModelManifest>) -> Result<CpuScalar> {
    resolve_backend_impl(reader, manifest, None)
}

pub fn resolve_profiled_backend(
    reader: &GgufReader,
    manifest: &ModelManifest,
    profile: &InferenceProfile,
) -> Result<CpuScalar> {
    resolve_backend_impl(reader, Some(manifest), Some(profile))
}

fn resolve_backend_impl(
    reader: &GgufReader,
    manifest: Option<&ModelManifest>,
    profile: Option<&InferenceProfile>,
) -> Result<CpuScalar> {
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
        if manifest.tensors.iter().any(|tensor| {
            let layout = &tensor.descriptor.physical_layout;
            layout != &request.tensor_layout && !(ggml_layout && layout == GGML_OUTPUT_ROWS_LAYOUT)
        }) {
            return Err(RuntimeError::InvalidInput(
                "model package declares an unsupported tensor layout".to_owned(),
            ));
        }
    }
    let mut plan = resolver.resolve(&request)?;
    let mixed = manifest.is_some_and(|manifest| manifest.quantization == QuantizationScheme::Mixed);
    let mut mixed_plans = if mixed {
        vec![plan.clone()]
    } else {
        Vec::new()
    };
    let mut projection_schemes = usize::from(
        mixed
            && reader.tensors().iter().any(|tensor| {
                tensor.name.starts_with("blk.")
                    && tensor.shape.len() == 2
                    && matches!(tensor.dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16)
            }),
    );

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
            request.tensor_layout = GGML_OUTPUT_ROWS_LAYOUT.to_owned();
        }
        let quantized_plan = resolver.resolve(&request)?;
        if quantized_plan.provider_name != plan.provider_name {
            return Err(RuntimeError::InvalidInput(
                "model tensors require incompatible CPU providers".to_owned(),
            ));
        }
        if mixed {
            mixed_plans.push(quantized_plan);
            projection_schemes += 1;
        } else if manifest.is_some_and(|manifest| {
            manifest.quantization == scheme
                || (profile.is_some() && manifest.quantization == QuantizationScheme::None)
        }) {
            plan = quantized_plan;
        }
    }

    if let Some(manifest) = manifest {
        if mixed && projection_schemes < 2 {
            return Err(RuntimeError::InvalidInput(
                "mixed precision package requires multiple projection tensor schemes".to_owned(),
            ));
        }
        if !manifest.execution_variants.is_empty()
            && !manifest.execution_variants.iter().any(|variant| {
                variant.runtime_provider == plan.provider_name
                    && variant.kernel_provider == plan.kernel_provider_name
                    && variant.quantization == manifest.quantization
            })
        {
            return Err(RuntimeError::InvalidInput(format!(
                "model package {} has no compatible CPU execution variant",
                manifest.model_id
            )));
        }
        if let Some(profile) = profile {
            if mixed {
                profile.bind_mixed(manifest, &mixed_plans)?;
            } else {
                profile.bind(manifest, &plan)?;
            }
        }
    }
    match plan.provider_name.as_str() {
        "cpu-scalar" => Ok(CpuScalar::with_config(CpuRuntimeConfig {
            threads: profile.map_or(1, |profile| profile.thread_count),
            ..CpuRuntimeConfig::default()
        })),
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

    #[test]
    fn binds_profile_to_the_manifest_variant_and_loaded_weight_scheme() {
        let mut writer = GgufWriter::new();
        let weights = quantinfer_core::QuantizedTensor::from_blocks(
            vec![32, 1],
            vec![Q8_0Block {
                scale: 0.5,
                values: [1; 32],
            }],
        )
        .unwrap();
        writer.add_q8_0_tensor("blk.0.attn_q.weight", &weights);
        let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
        let mut manifest = ModelManifest::new("fixture", "qwen3");
        manifest.quantization = QuantizationScheme::Q8_0;
        manifest.execution_variants.push(ExecutionVariant {
            name: "q8-cpu".to_owned(),
            quantization: QuantizationScheme::Q8_0,
            dtype: "f32-accumulation".to_owned(),
            runtime_provider: "cpu-scalar".to_owned(),
            kernel_provider: "cpu-scalar".to_owned(),
        });
        let mut profile = InferenceProfile {
            model_variant: "q8-cpu".to_owned(),
            quantization_variant: "q8_0".to_owned(),
            runtime_provider: "cpu-scalar".to_owned(),
            kernel_provider: "cpu-scalar".to_owned(),
            context_length: 64,
            thread_count: 2,
            memory_budget_bytes: Some(1 << 20),
            quality_threshold: None,
            performance_baseline_tokens_per_second: None,
        };
        let backend = resolve_profiled_backend(&reader, &manifest, &profile).unwrap();
        assert_eq!(backend.config().threads, 2);
        profile.model_variant = "other".to_owned();
        assert!(resolve_profiled_backend(&reader, &manifest, &profile).is_err());
        profile.model_variant = "q8-cpu".to_owned();
        profile.quantization_variant = "f32".to_owned();
        manifest.quantization = QuantizationScheme::None;
        manifest.execution_variants[0].quantization = QuantizationScheme::None;
        assert!(resolve_profiled_backend(&reader, &manifest, &profile).is_err());
    }
}
