//! Validated offline GPTQ workflow configuration.

use crate::{
    common::{optional_value, required_path, validate_options},
    convert::qif_options::QifOptions,
};
use quantinfer_adapter_gptq::{GptqProvider, GptqQuantizationOptions};
use std::{collections::BTreeSet, path::PathBuf};

pub(super) struct WorkflowOptions {
    pub input: PathBuf,
    pub dataset: PathBuf,
    pub validation: Option<PathBuf>,
    pub output: PathBuf,
    pub report: Option<PathBuf>,
    pub model_id: Option<String>,
    pub context: Option<usize>,
    pub statistics_bytes: usize,
    pub dataset_bytes: usize,
    pub maximum_kl: f64,
    pub minimum_top10: usize,
    pub maximum_validation_prompts: usize,
    pub provider: GptqProvider,
    pub qif: QifOptions,
}

impl WorkflowOptions {
    pub fn parse(arguments: &[String]) -> Result<Self, String> {
        let options = [
            ("--input", true),
            ("--dataset", true),
            ("--validation-dataset", true),
            ("--output", true),
            ("--report", true),
            ("--model-id", true),
            ("--max-sequence-length", true),
            ("--bits", true),
            ("--group-size", true),
            ("--damping", true),
            ("--activation-order", false),
            ("--maximum-workspace-bytes", true),
            ("--maximum-statistics-bytes", true),
            ("--max-dataset-bytes", true),
            ("--maximum-kl", true),
            ("--minimum-top10-overlap", true),
            ("--max-validation-prompts", true),
            ("--qif-profile", true),
            ("--qif-codec", true),
            ("--max-shard-bytes", true),
        ];
        validate_options(arguments, &options)?;
        let mut seen = BTreeSet::new();
        let mut index = 0;
        while index < arguments.len() {
            let flag = arguments[index].as_str();
            if !seen.insert(flag) {
                return Err(format!("duplicate option {flag}"));
            }
            index += 1 + usize::from(options.iter().find(|(name, _)| *name == flag).unwrap().1);
        }
        let integer = |flag: &str, default: usize| -> Result<usize, String> {
            let value = optional_value(arguments, flag)
                .map(|value| {
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("{flag} must be an integer"))
                })
                .transpose()?
                .unwrap_or(default);
            if value == 0 {
                return Err(format!("{flag} must be positive"));
            }
            Ok(value)
        };
        let number = |flag: &str, default: f64| -> Result<f64, String> {
            let value = optional_value(arguments, flag)
                .map(|v| {
                    v.parse::<f64>()
                        .map_err(|_| format!("{flag} must be a number"))
                })
                .transpose()?
                .unwrap_or(default);
            if !value.is_finite() || value <= 0. {
                return Err(format!("{flag} must be finite and positive"));
            }
            Ok(value)
        };
        let defaults = GptqQuantizationOptions::default();
        let provider = GptqProvider::new(GptqQuantizationOptions {
            bits: integer("--bits", defaults.bits)?,
            group_size: integer("--group-size", defaults.group_size)?,
            damping: number("--damping", defaults.damping)?,
            activation_order: seen.contains("--activation-order"),
            maximum_workspace_bytes: integer(
                "--maximum-workspace-bytes",
                defaults.maximum_workspace_bytes,
            )?,
        })
        .map_err(|error| error.to_string())?;
        let output = required_path(arguments, "--output")?;
        let validation = optional_value(arguments, "--validation-dataset").map(PathBuf::from);
        if validation.is_none()
            && (seen.contains("--maximum-kl") || seen.contains("--minimum-top10-overlap"))
        {
            return Err("quality thresholds require --validation-dataset".into());
        }
        let minimum_top10 = optional_value(arguments, "--minimum-top10-overlap")
            .map(|v| {
                v.parse::<usize>()
                    .map_err(|_| "--minimum-top10-overlap must be an integer".to_owned())
            })
            .transpose()?
            .unwrap_or(6);
        if minimum_top10 > 10 {
            return Err("--minimum-top10-overlap must be between 0 and 10".into());
        }
        let report = optional_value(arguments, "--report").map(PathBuf::from);
        let model_id = optional_value(arguments, "--model-id");
        if model_id.as_ref().is_some_and(String::is_empty) {
            return Err("--model-id must not be empty".into());
        }
        let input = required_path(arguments, "--input")?;
        let dataset = required_path(arguments, "--dataset")?;
        for path in [&input, &dataset, &output]
            .into_iter()
            .chain(validation.as_ref())
            .chain(report.as_ref())
        {
            if path.as_os_str().is_empty() {
                return Err("workflow file paths must not be empty".into());
            }
        }
        for path in std::iter::once(&output).chain(report.as_ref()) {
            match std::fs::symlink_metadata(path) {
                Ok(_) => return Err(format!("output is already occupied at {}", path.display())),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(format!("cannot inspect output {}: {error}", path.display()))
                }
            }
        }
        if report.as_ref().is_some_and(|path| path == &output) {
            return Err("--report and --output must be different paths".into());
        }
        Ok(Self {
            input,
            dataset,
            validation,
            output,
            report,
            model_id,
            context: optional_value(arguments, "--max-sequence-length")
                .map(|_| integer("--max-sequence-length", 512))
                .transpose()?,
            statistics_bytes: integer("--maximum-statistics-bytes", 512 * 1024 * 1024)?,
            dataset_bytes: integer("--max-dataset-bytes", 32 * 1024 * 1024)?,
            maximum_kl: number("--maximum-kl", 0.5)?,
            provider,
            qif: QifOptions::parse(arguments)?,
            minimum_top10,
            maximum_validation_prompts: integer("--max-validation-prompts", 16)?,
        })
    }
}
