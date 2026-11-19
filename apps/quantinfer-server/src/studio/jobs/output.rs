//! Verify staged artifacts and read bounded structured reports.

use std::fs;
use std::io::Read;
use std::path::Path;

use quantinfer::format::PackedTensorReader;
use serde_json::Value;

use super::io_error;

#[derive(Clone, Copy)]
pub(super) enum Output {
    Model,
    Calibration,
    Benchmark,
    Package,
    Verification,
}

impl Output {
    pub(super) fn arguments(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::Model => &[("--output", "model.qif"), ("--report", "quantization.json")],
            Self::Calibration => &[("--output", "imatrix.json")],
            Self::Package => &[("--output", "model.qif")],
            Self::Benchmark => &[("--profile-report", "benchmark.json")],
            Self::Verification => &[],
        }
    }

    pub(super) fn verify(self, stage: &Path) -> Result<Value, String> {
        match self {
            Self::Package => {
                let path = stage.join("model.qif");
                let mut reader =
                    quantinfer::format::qif::PackageReader::open(&path).map_err(io_error)?;
                reader.verify().map_err(io_error)?;
                Ok(
                    serde_json::json!({"artifact": "model.qif", "profile": "PACKAGE", "verified": true, "tensor_count": reader.tensors().len()}),
                )
            }
            Self::Verification => Ok(serde_json::json!({"verified": true})),
            Self::Model => {
                let reader =
                    PackedTensorReader::open_path(stage.join("model.qif")).map_err(io_error)?;
                reader.verify_qif().map_err(io_error)?;
                let report = read_json(&stage.join("quantization.json"), 64 * 1024 * 1024)?;
                if report["artifact_parity"] != true || report["payload_verified"] != true {
                    return Err("converter did not report verified model parity".into());
                }
                Ok(
                    serde_json::json!({"artifact": "model.qif", "report": "quantization.json", "quality_status": report["quality_status"]}),
                )
            }
            Self::Calibration => {
                let report = read_json(&stage.join("imatrix.json"), 64 * 1024 * 1024)?;
                let tensors = report
                    .as_object()
                    .filter(|map| !map.is_empty())
                    .ok_or("empty calibration report")?;
                Ok(serde_json::json!({"artifact": "imatrix.json", "tensor_count": tensors.len()}))
            }
            Self::Benchmark => {
                let report = read_json(&stage.join("benchmark.json"), 64 * 1024)?;
                let metrics = &report["metrics"];
                if report["schema_version"] != 1
                    || ["model_load_ms", "prefill_ms", "ttft_ms", "decode_tokens"]
                        .iter()
                        .any(|field| {
                            metrics[*field]
                                .as_f64()
                                .is_none_or(|number| !number.is_finite() || number < 0.0)
                        })
                {
                    return Err("benchmark did not report its required metrics".into());
                }
                Ok(report)
            }
        }
    }
}

pub(in crate::studio) fn read_json(path: &Path, limit: u64) -> Result<Value, String> {
    let file = fs::File::open(path).map_err(io_error)?;
    if !file.metadata().map_err(io_error)?.is_file() {
        return Err("Studio reports must be regular files".into());
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() as u64 > limit {
        return Err("Studio report exceeds the read limit".into());
    }
    serde_json::from_slice(&bytes).map_err(io_error)
}
