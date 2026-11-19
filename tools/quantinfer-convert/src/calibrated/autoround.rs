//! Per-projection SignRound tuning on the activations of the updated model.

use super::{
    execute, layer, model, options, packing, validation, ModelQuantizer, Production,
    ProductionContext,
};
use crate::common::optional_value;
use quantinfer::{model::TensorRole, runtime::dense_storage_name};
use quantinfer_adapter_autoround::{AutoRoundCalibration, AutoRoundOptions, AutoRoundProvider};
use quantinfer_core::{QuantizationScheme, Tensor};
use quantinfer_quant::QuantizationProvider;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use TensorRole::*;

pub(crate) fn run(arguments: &[String]) -> Result<(), String> {
    let workflow = options::WorkflowOptions::parse(
        arguments,
        &[
            ("--bits", true),
            ("--group-size", true),
            ("--iterations", true),
            ("--rounding-lr", true),
            ("--clipping-lr", true),
            ("--threads", true),
            ("--maximum-workspace-bytes", true),
        ],
    )?;
    let defaults = AutoRoundOptions::default();
    let step = |name| {
        optional_value(arguments, name)
            .map(|_| options::number(arguments, name, 0.01))
            .transpose()
    };
    let provider = AutoRoundProvider::new(AutoRoundOptions {
        bits: options::integer(arguments, "--bits", defaults.bits)?,
        group_size: options::integer(arguments, "--group-size", defaults.group_size)?,
        iterations: options::integer(arguments, "--iterations", defaults.iterations)?,
        rounding_lr: step("--rounding-lr")?,
        clipping_lr: step("--clipping-lr")?,
        threads: options::integer(arguments, "--threads", defaults.threads)?,
        maximum_workspace_bytes: options::integer(
            arguments,
            "--maximum-workspace-bytes",
            defaults.maximum_workspace_bytes,
        )?,
    })
    .map_err(|e| e.to_string())?;
    execute(workflow, &Workflow(provider))
}

struct Workflow(AutoRoundProvider);

impl ModelQuantizer for Workflow {
    fn name(&self) -> &'static str {
        "autoround"
    }
    fn id(&self) -> &'static str {
        "autoround-signed-f16"
    }
    fn bits(&self) -> usize {
        self.0.options().bits
    }
    fn scheme(&self) -> QuantizationScheme {
        self.0.scheme()
    }
    fn parameters(&self) -> Value {
        let o = self.0.options();
        let default_step = 1.0 / o.iterations as f64;
        json!({"bits":o.bits, "group_size":o.group_size, "iterations":o.iterations,
            "rounding_lr": o.rounding_lr.unwrap_or(default_step),
            "clipping_lr": o.clipping_lr.unwrap_or(default_step),
            "threads":o.threads})
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
        let mut hidden = layer::embed_prompts(model, spec, prompts, statistics_bytes)?;
        let mut publisher = Publisher {
            provider: &self.0,
            tensors: manifest
                .tensors
                .iter_mut()
                .map(|t| (t.name.clone(), t))
                .collect(),
            payloads: BTreeMap::new(),
            records: Vec::new(),
            ggml_llama: ggml && spec.architecture == "llama",
            statistics_bytes,
        };
        for (index, layer_spec) in spec.layers.iter().enumerate() {
            let mut layer = layer::Layer {
                model,
                index,
                context,
            };
            let attention = layer.normalized(InputNormalization, &hidden)?;
            publisher.group(
                &mut layer,
                model::GROUPS[0],
                &attention,
                layer_spec.attention.head_dim,
            )?;
            let output = layer.observe(&attention, OutputProjection)?;
            publisher.group(
                &mut layer,
                model::GROUPS[1],
                &output,
                layer_spec.attention.head_dim,
            )?;
            drop(output);

            let residual = layer.ffn_residual(&hidden, &attention)?;
            let feed_forward = layer.normalized(PostAttentionNormalization, &residual)?;
            publisher.group(
                &mut layer,
                model::GROUPS[2],
                &feed_forward,
                layer_spec.attention.head_dim,
            )?;
            let down = layer.observe(&feed_forward, DownProjection)?;
            publisher.group(
                &mut layer,
                model::GROUPS[3],
                &down,
                layer_spec.attention.head_dim,
            )?;
            drop(down);
            if index + 1 < spec.layers.len() {
                layer.advance(&mut hidden)?;
            }
            println!("autoround_layer={}", index + 1);
        }
        Ok(Production {
            payloads: publisher.payloads,
            records: publisher.records,
            details: BTreeMap::new(),
        })
    }
}

struct Publisher<'a> {
    provider: &'a AutoRoundProvider,
    tensors: BTreeMap<String, &'a mut quantinfer_format::TensorManifest>,
    payloads: BTreeMap<String, Vec<u8>>,
    records: Vec<Value>,
    ggml_llama: bool,
    statistics_bytes: usize,
}

impl Publisher<'_> {
    fn group(
        &mut self,
        layer: &mut layer::Layer<'_>,
        roles: &[TensorRole],
        samples: &[Tensor],
        head_dim: usize,
    ) -> Result<(), String> {
        let width = layer.weight(roles[0])?.shape().dims()[0];
        let mut calibration =
            AutoRoundCalibration::new(width, self.statistics_bytes).map_err(|e| e.to_string())?;
        for input in samples {
            calibration.observe_rows(input).map_err(|e| e.to_string())?;
        }
        for &role in roles {
            let name = dense_storage_name(role, Some(layer.index));
            let weight = layer.weight(role)?;
            let result = self
                .provider
                .optimize(weight, &calibration)
                .map_err(|e| format!("cannot optimize {name}: {e}"))?;
            let error = self
                .provider
                .error_report(weight, &result.weights)
                .map_err(|e| e.to_string())?;
            let rope = (self.ggml_llama && matches!(role, QueryProjection | KeyProjection))
                .then_some(head_dim);
            let raw = packing::encoded_projection(&result.weights, &name, rope)?;
            let tensor = self
                .tensors
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
            self.records.push(json!({"tensor":name,"layer":layer.index,"role":format!("{role:?}"),
                "calibration_rows":calibration.rows(),"source_core_f32_sha256":validation::tensor_hash(weight)?,
                "payload_sha256":quantinfer_format::sha256_hex(&raw),
                "initial_mse":result.initial_mse,"best_mse":result.best_mse,
                "improved_outputs":result.improved_outputs,
                "maximum_weight_error":error.max_absolute_error,"mean_weight_error":error.mean_absolute_error,
                "relative_l2":error.relative_l2_error}));
            self.payloads.insert(name, raw);
            model::replace_projection(
                layer.model,
                layer.index,
                role,
                packing::runtime_weight(result.weights)?,
            )?;
        }
        Ok(())
    }
}
