//! Full-Hessian GPTQ production, independent from source and publication policy.
use super::{model, packing, validation, Production, ProductionContext};
use quantinfer::{
    model::TensorRole,
    runtime::{dense_storage_name, LinearWeight},
};
use quantinfer_adapter_gptq::{GptqProvider, GptqQuantizationOptions, HessianCalibration};
use quantinfer_quant::{CalibratedQuantizationProvider, QuantizationProvider};
use serde_json::json;
use std::collections::BTreeMap;

pub(super) fn produce(
    context: ProductionContext<'_>,
    provider: &GptqProvider,
) -> Result<Production, String> {
    let ProductionContext {
        model,
        manifest,
        spec,
        prompts,
        context,
        ggml,
        statistics_bytes,
    } = context;
    let mut payloads = BTreeMap::<String, Vec<u8>>::new();
    let mut records = Vec::new();
    for layer in 0..spec.layers.len() {
        for roles in model::GROUPS {
            let input = model::projection(model, layer, roles[0])?.shape().dims()[0];
            let mut calibration =
                HessianCalibration::new(input, statistics_bytes).map_err(|e| e.to_string())?;
            for tokens in prompts {
                let mut observer =
                    |observed_layer,
                     observed_roles: &[TensorRole],
                     activations: &quantinfer_core::Tensor| {
                        if observed_layer == layer
                            && observed_roles.iter().any(|role| roles.contains(role))
                        {
                            calibration.observe_rows(activations).map_err(|e| {
                                quantinfer::runtime::RuntimeError::InvalidInput(e.to_string())
                            })?;
                        }
                        Ok(())
                    };
                model
                    .observe_projections(
                        tokens,
                        &mut model.new_cache(context).map_err(|e| e.to_string())?,
                        &mut observer,
                    )
                    .map_err(|e| e.to_string())?;
            }
            for &role in roles {
                let name = dense_storage_name(role, Some(layer));
                let LinearWeight::F32(weights) = model::projection(model, layer, role)? else {
                    return Err(format!("{name} is not a floating projection"));
                };
                let quantized = provider
                    .quantize_calibrated(weights, &calibration)
                    .map_err(|e| format!("cannot quantize {name}: {e}"))?;
                let error = provider
                    .error_report(weights, &quantized)
                    .map_err(|e| e.to_string())?;
                let rope_dim = (ggml
                    && spec.architecture == "llama"
                    && matches!(
                        role,
                        TensorRole::QueryProjection | TensorRole::KeyProjection
                    ))
                .then_some(spec.layers[layer].attention.head_dim);
                let raw = packing::encoded_projection(&quantized, &name, rope_dim)?;
                let tensor = manifest
                    .tensors
                    .iter_mut()
                    .find(|t| t.name == name)
                    .ok_or_else(|| format!("manifest is missing {name}"))?;
                let mut descriptor = provider
                    .storage_descriptor(weights.shape().clone())
                    .map_err(|e| e.to_string())?;
                descriptor.logical_shape = tensor.descriptor.logical_shape.clone();
                tensor.descriptor = descriptor;
                tensor.checksum_sha256 = None;
                records.push(json!({"tensor":name,"layer":layer,"role":format!("{role:?}"),"calibration_rows":calibration.rows(),
                    "source_core_f32_sha256":validation::tensor_hash(weights)?,"payload_sha256":quantinfer_format::sha256_hex(&raw),
                    "maximum_weight_error":error.max_absolute_error,"mean_weight_error":error.mean_absolute_error,"relative_l2":error.relative_l2_error}));
                payloads.insert(name.clone(), raw);
                model::replace_projection(model, layer, role, packing::runtime_weight(quantized)?)?;
                println!("gptq_projection={name}");
            }
        }
    }
    Ok(Production {
        payloads,
        records,
        details: BTreeMap::new(),
    })
}

const OPTIONS: &[(&str, bool)] = &[
    ("--bits", true),
    ("--group-size", true),
    ("--damping", true),
    ("--activation-order", false),
    ("--maximum-workspace-bytes", true),
];

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    use super::options::{integer, number, WorkflowOptions};
    let options = WorkflowOptions::parse(arguments, OPTIONS)?;
    let d = GptqQuantizationOptions::default();
    let provider = GptqProvider::new(GptqQuantizationOptions {
        bits: integer(arguments, "--bits", d.bits)?,
        group_size: integer(arguments, "--group-size", d.group_size)?,
        damping: number(arguments, "--damping", d.damping)?,
        activation_order: arguments.iter().any(|a| a == "--activation-order"),
        maximum_workspace_bytes: integer(
            arguments,
            "--maximum-workspace-bytes",
            d.maximum_workspace_bytes,
        )?,
    })
    .map_err(|e| e.to_string())?;
    super::execute(options, &Workflow(provider))
}

struct Workflow(GptqProvider);
impl super::ModelQuantizer for Workflow {
    fn name(&self) -> &'static str {
        "gptq"
    }
    fn id(&self) -> &'static str {
        "gptq-static-signed-f16"
    }
    fn bits(&self) -> usize {
        self.0.options().bits
    }
    fn scheme(&self) -> quantinfer_core::QuantizationScheme {
        self.0.scheme()
    }
    fn parameters(&self) -> serde_json::Value {
        let o = self.0.options();
        json!({"bits":o.bits,"group_size":o.group_size,"damping":o.damping,"activation_order":o.activation_order})
    }
    fn produce(&self, context: ProductionContext<'_>) -> Result<Production, String> {
        produce(context, &self.0)
    }
}
