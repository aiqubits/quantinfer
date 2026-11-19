//! Layerwise AWQ for the native signed QIF weight profiles.

mod layer;

use super::layer as execution;

use super::{
    execute, model, options, packing, validation, ModelQuantizer, Production, ProductionContext,
};
use quantinfer::{model::TensorRole, runtime::dense_storage_name};
use quantinfer_adapter_awq::{AwqOptions, AwqProvider};
use quantinfer_core::{DType, QuantizationScheme, TensorStorageDescriptor};
use quantinfer_quant::QuantizationProvider;
use serde_json::{json, Value};
use std::collections::BTreeMap;

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    let config = options::WorkflowOptions::parse(
        arguments,
        &[
            ("--bits", true),
            ("--group-size", true),
            ("--grid-points", true),
            ("--maximum-workspace-bytes", true),
            ("--no-clipping", false),
        ],
    )?;
    let defaults = AwqOptions::default();
    let provider = AwqProvider::new(AwqOptions {
        bits: options::integer(arguments, "--bits", defaults.bits)?,
        group_size: options::integer(arguments, "--group-size", defaults.group_size)?,
        grid_points: options::integer(arguments, "--grid-points", defaults.grid_points)?,
        maximum_workspace_bytes: options::integer(
            arguments,
            "--maximum-workspace-bytes",
            defaults.maximum_workspace_bytes,
        )?,
    })
    .map_err(|e| e.to_string())?;
    execute(
        config,
        &Workflow {
            provider,
            clipping: !arguments.iter().any(|a| a == "--no-clipping"),
        },
    )
}

struct Workflow {
    provider: AwqProvider,
    clipping: bool,
}

impl ModelQuantizer for Workflow {
    fn name(&self) -> &'static str {
        "awq"
    }
    fn id(&self) -> &'static str {
        "awq-signed-f16"
    }
    fn bits(&self) -> usize {
        self.provider.options().bits
    }
    fn scheme(&self) -> QuantizationScheme {
        self.provider.scheme()
    }

    fn parameters(&self) -> Value {
        let o = self.provider.options();
        json!({"bits": o.bits, "group_size": o.group_size, "grid_points": o.grid_points, "clipping": self.clipping})
    }

    fn produce(&self, ctx: ProductionContext<'_>) -> Result<Production, String> {
        let ProductionContext {
            model,
            manifest,
            spec,
            prompts,
            context,
            ggml,
            statistics_bytes,
        } = ctx;
        let mut hidden = execution::embed_prompts(model, spec, prompts, statistics_bytes)?;
        let mut payloads = BTreeMap::new();
        let mut records = Vec::new();
        let mut norms = Vec::new();
        let mut scales = Vec::new();
        let mut clipping = Vec::new();
        let mut tensors: BTreeMap<_, _> = manifest
            .tensors
            .iter_mut()
            .map(|t| (t.name.clone(), t))
            .collect();
        for index in 0..spec.layers.len() {
            let mut layer = execution::Layer {
                model,
                index,
                context,
            };
            let mut source_hashes = BTreeMap::new();
            for role in model::GROUPS.into_iter().flatten().copied() {
                source_hashes.insert(role, validation::tensor_hash(layer.weight(role)?)?);
            }
            scales.extend(layer.scale(&self.provider, &hidden)?);
            if self.clipping {
                clipping.extend(layer.clip(&self.provider, &hidden)?);
            }

            for role in [
                TensorRole::InputNormalization,
                TensorRole::PostAttentionNormalization,
            ] {
                let name = dense_storage_name(role, Some(index));
                let tensor = tensors
                    .get_mut(&name)
                    .ok_or_else(|| format!("manifest missing {name}"))?;
                let raw: Vec<u8> = layer
                    .norm(role)?
                    .as_f32()
                    .map_err(|e| e.to_string())?
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect();
                norms.push(
                    json!({"tensor":name, "source_dtype":tensor.descriptor.dtype,
                    "payload_sha256":quantinfer_format::sha256_hex(&raw)}),
                );
                let mut descriptor = TensorStorageDescriptor::floating(
                    DType::F32,
                    tensor.descriptor.logical_shape.clone(),
                )
                .map_err(|e| e.to_string())?;
                descriptor.backend_requirements =
                    std::mem::take(&mut tensor.descriptor.backend_requirements);
                tensor.descriptor = descriptor;
                tensor.checksum_sha256 = None;
                payloads.insert(name, raw);
            }

            for role in model::GROUPS.into_iter().flatten().copied() {
                let name = dense_storage_name(role, Some(index));
                let weight = layer.weight(role)?;
                let packed = self
                    .provider
                    .quantize_transformed(weight)
                    .map_err(|e| format!("cannot pack {name}: {e}"))?;
                let errors = self
                    .provider
                    .error_report(weight, &packed)
                    .map_err(|e| e.to_string())?;
                let rope = (ggml
                    && spec.architecture == "llama"
                    && matches!(
                        role,
                        TensorRole::QueryProjection | TensorRole::KeyProjection
                    ))
                .then_some(spec.layers[index].attention.head_dim);
                let raw = packing::encoded_projection(&packed, &name, rope)?;
                let tensor = tensors
                    .get_mut(&name)
                    .ok_or_else(|| format!("manifest missing {name}"))?;
                let mut descriptor = self
                    .provider
                    .storage_descriptor(weight.shape().clone())
                    .map_err(|e| e.to_string())?;
                descriptor.logical_shape = tensor.descriptor.logical_shape.clone();
                descriptor.backend_requirements =
                    std::mem::take(&mut tensor.descriptor.backend_requirements);
                tensor.descriptor = descriptor;
                tensor.checksum_sha256 = None;
                records.push(
                    json!({"tensor":name,"layer":index,"role":format!("{role:?}"),
                    "source_core_f32_sha256":source_hashes[&role],
                    "transformed_core_f32_sha256":validation::tensor_hash(weight)?,
                    "payload_sha256":quantinfer_format::sha256_hex(&raw),
                    "transformed_maximum_weight_error":errors.max_absolute_error,
                    "transformed_mean_weight_error":errors.mean_absolute_error,
                    "transformed_relative_l2":errors.relative_l2_error}),
                );
                payloads.insert(name, raw);
                model::replace_projection(
                    layer.model,
                    index,
                    role,
                    packing::runtime_weight(packed)?,
                )?;
            }
            if index + 1 < spec.layers.len() {
                layer.advance(&mut hidden)?;
            }
            println!("awq_layer={}/{}", index + 1, spec.layers.len());
        }
        Ok(Production {
            payloads,
            records,
            details: BTreeMap::from([
                ("normalizations".into(), json!(norms)),
                ("scale_searches".into(), json!(scales)),
                ("clipping_searches".into(), json!(clipping)),
            ]),
        })
    }
}
