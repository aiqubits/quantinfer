//! Measured, budget-constrained precision planning for native dense conversion.
mod options;

pub(crate) use options::OptimizationOptions;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::Read,
    path::Path,
};

use quantinfer_core::{QuantizationScheme, Tensor};
use quantinfer_format::{
    canonical_dense_tensor_name, sha256_hex, sha256_reader, transpose_dense_layout, GgufValue,
    GgufWriter,
};
use quantinfer_quant::{
    measure_linear_precision_candidates, optimize_mixed_precision, MixedPrecisionPlan,
    PrecisionSearchLimits,
};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{
    common::{is_quantizable_linear, load_imatrix},
    policy::parse_quantization,
    source::WeightSource,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Policy {
    maximum_encoded_weight_bytes: u64,
    candidates: Vec<String>,
    importance_weights: Option<String>,
    max_frontier_states: Option<usize>,
    max_history_states: Option<usize>,
}

pub(crate) struct OptimizedConversion {
    pub plan: MixedPrecisionPlan,
    pub variant: QuantizationScheme,
    policy_sha256: String,
    importance_sha256: Option<String>,
    input_hashes: BTreeMap<String, String>,
    report: Value,
}

fn input_hash(values: &[f32]) -> String {
    sha256_hex(
        &values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

impl OptimizedConversion {
    pub fn prepare(weights: &WeightSource, path: &Path) -> Result<Self, String> {
        let mut bytes = Vec::new();
        File::open(path)
            .and_then(|file| file.take(65_537).read_to_end(&mut bytes))
            .map_err(|error| format!("cannot read precision policy {}: {error}", path.display()))?;
        if bytes.len() > 65_536 {
            return Err("precision policy exceeds 64 KiB".into());
        }
        let policy: Policy =
            serde_json::from_slice(&bytes).map_err(|e| format!("invalid precision policy: {e}"))?;
        let policy_sha256 = sha256_hex(&bytes);
        let schemes = policy
            .candidates
            .iter()
            .map(|s| parse_quantization(s))
            .collect::<Result<Vec<_>, _>>()?;
        let (mut importance, importance_sha256) = if let Some(name) = policy.importance_weights {
            let path = path.parent().unwrap_or(Path::new(".")).join(name);
            let bytes = fs::read(&path)
                .map_err(|e| format!("cannot read importance weights {}: {e}", path.display()))?;
            let digest = sha256_hex(&bytes);
            (Some(load_imatrix(bytes)?), Some(digest))
        } else {
            (None, None)
        };
        let mut fixed = 0_u64;
        let mut groups = Vec::new();
        let mut reports = Vec::new();
        let mut names = BTreeSet::new();
        let mut input_hashes = BTreeMap::new();
        for (source_name, shape) in weights.tensors()? {
            let name =
                canonical_dense_tensor_name(&source_name).unwrap_or_else(|| source_name.clone());
            if !names.insert(name.clone()) {
                return Err(format!("multiple source entries map to {name}"));
            }
            if !is_quantizable_linear(&name) {
                let elements = shape
                    .iter()
                    .try_fold(1_u64, |n, &d| n.checked_mul(d as u64))
                    .ok_or("fixed precision weight shape overflows")?;
                fixed = elements
                    .checked_mul(4)
                    .and_then(|n| fixed.checked_add(n))
                    .ok_or("fixed precision byte count overflows")?;
                continue;
            }
            let values = weights
                .tensor_f32(&source_name)
                .map_err(|e| e.to_string())?;
            let (shape, values) = transpose_dense_layout(&name, &shape, values)?;
            input_hashes.insert(name.clone(), input_hash(&values));
            let channel_weights = importance
                .as_mut()
                .map(|weights| {
                    weights
                        .remove(&name)
                        .ok_or_else(|| format!("importance weights missing {name}"))
                })
                .transpose()?;
            let tensor = Tensor::from_f32(shape.clone(), values).map_err(|e| e.to_string())?;
            let measured = measure_linear_precision_candidates(
                &name,
                &tensor,
                &schemes,
                channel_weights.as_deref(),
            )
            .map_err(|e| format!("measure {name}: {e}"))?;
            let candidates = measured.group.candidates.iter().map(|c|json!({
                "scheme":c.scheme.name(),"encoded_bytes":c.encoded_bytes,"weighted_sse":c.error
            })).collect::<Vec<_>>();
            let unavailable = measured
                .unavailable
                .iter()
                .map(|c| {
                    json!({
                        "scheme":c.scheme.name(),"reason":c.reason
                    })
                })
                .collect::<Vec<_>>();
            reports.push(json!({"name":name,"shape":shape,"candidates":candidates,"unavailable":unavailable}));
            groups.push(measured.group);
        }
        if let Some(name) = importance
            .as_ref()
            .and_then(|weights| weights.keys().next())
        {
            return Err(format!("unused importance tensor {name}"));
        }
        let defaults = PrecisionSearchLimits::default();
        let limits = PrecisionSearchLimits {
            max_frontier_states: policy
                .max_frontier_states
                .unwrap_or(defaults.max_frontier_states),
            max_history_states: policy
                .max_history_states
                .unwrap_or(defaults.max_history_states),
        };
        let solved =
            optimize_mixed_precision(&groups, fixed, policy.maximum_encoded_weight_bytes, limits)
                .map_err(|e| e.to_string())?;
        let mut selected_schemes = Vec::new();
        for (report, group) in reports.iter_mut().zip(&groups) {
            let scheme = solved
                .plan
                .scheme_for(&group.tensor_names[0])
                .expect("solved every group");
            let choice = group
                .candidates
                .iter()
                .find(|c| c.scheme == scheme)
                .unwrap();
            report["selected"] = json!({"scheme":scheme.name(),"encoded_bytes":choice.encoded_bytes,"weighted_sse":choice.error});
            report["input_f32_sha256"] = json!(input_hashes[&group.tensor_names[0]]);
            if !selected_schemes.contains(&scheme) {
                selected_schemes.push(scheme);
            }
        }
        let variant = if selected_schemes.len() == 1 {
            selected_schemes[0]
        } else {
            QuantizationScheme::Mixed
        };
        let report = json!({
            "policy_sha256":policy_sha256,"importance_sha256":importance_sha256,
            "budget_scope":"sum_of_logical_encoded_tensor_payloads_excludes_alias_deduplication_padding_compression_metadata_and_runtime_memory",
            "objective":if importance.is_some() {"diagonal_activation_weighted_sse"} else {"unweighted_reconstruction_sse"},
            "fixed_encoded_weight_bytes":fixed,"maximum_encoded_weight_bytes":policy.maximum_encoded_weight_bytes,
            "selected_encoded_weight_bytes":solved.encoded_bytes,"total_weighted_sse":solved.error,
            "search_limits":{"max_frontier_states":limits.max_frontier_states,"max_history_states":limits.max_history_states},
            "variant":variant.name(),"tensors":reports
        });
        Ok(Self {
            plan: solved.plan,
            variant,
            policy_sha256,
            importance_sha256,
            input_hashes,
            report,
        })
    }

    pub fn metadata(&self, writer: &mut GgufWriter) {
        writer.insert_metadata(
            "quantinfer.quantization.variant",
            GgufValue::String(self.variant.name().into()),
        );
        writer.insert_metadata(
            "quantinfer.quantization.optimization_policy_sha256",
            GgufValue::String(self.policy_sha256.clone()),
        );
        if let Some(hash) = &self.importance_sha256 {
            writer.insert_metadata(
                "quantinfer.quantization.optimization_imatrix_sha256",
                GgufValue::String(hash.clone()),
            );
        }
    }

    pub fn validate_input(&self, name: &str, values: &[f32]) -> Result<(), String> {
        if let Some(expected) = self.input_hashes.get(name) {
            if expected != &input_hash(values) {
                return Err(format!("tensor {name} changed after precision planning"));
            }
        }
        Ok(())
    }

    pub fn write_report(&self, path: &Path, artifact: &Path) -> Result<(), String> {
        let files = if artifact.is_dir() {
            fs::read_dir(artifact)
                .map_err(|e| e.to_string())?
                .map(|entry| entry.map(|e| e.path()))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| e.to_string())?
        } else {
            vec![artifact.to_path_buf()]
        };
        let mut hashes = BTreeMap::new();
        for file in files {
            let hash = File::open(&file)
                .and_then(|mut f| sha256_reader(&mut f))
                .map_err(|e| e.to_string())?;
            hashes.insert(
                file.file_name().unwrap().to_string_lossy().into_owned(),
                hash,
            );
        }
        let mut report = self.report.clone();
        report["artifact_sha256"] = json!(hashes);
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let temporary = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        fs::write(
            temporary.path(),
            serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        temporary
            .persist_noclobber(path)
            .map_err(|e| format!("cannot publish precision report: {e}"))?;
        Ok(())
    }
}
