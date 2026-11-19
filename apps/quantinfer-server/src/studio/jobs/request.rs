//! Validated choices and fixed argument lists for the local tools.

use std::fs;
use std::io::Read;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{child, io_error, output::Output, Jobs};

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Algorithm {
    Gptq,
    Awq,
    Autoround,
}

impl Algorithm {
    fn command(self) -> &'static str {
        match self {
            Self::Gptq => "gptq",
            Self::Awq => "awq",
            Self::Autoround => "autoround",
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Codec {
    None,
    Zstd,
    Lz4,
}

impl Codec {
    fn argument(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Zstd => "zstd",
            Self::Lz4 => "lz4",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Request {
    Quantize {
        source: String,
        dataset: String,
        validation_dataset: Option<String>,
        algorithm: Algorithm,
        bits: usize,
        group_size: usize,
        context: usize,
        package: bool,
        codec: Codec,
    },
    Calibrate {
        source: String,
        dataset: String,
        context: usize,
    },
    Package {
        source: String,
        codec: Codec,
        max_shard_bytes: Option<usize>,
    },
    Verify {
        source: String,
    },
    Benchmark {
        source: String,
        prompt: String,
        tokens: usize,
        context: usize,
    },
}

pub(super) struct Prepared {
    pub(super) executable: PathBuf,
    pub(super) arguments: Vec<String>,
    pub(super) output: Output,
}

impl Jobs {
    pub(super) fn prepare(&self, request: &Request) -> Result<Prepared, String> {
        let source = |name: &str| self.source(name);
        let dataset = |name: &str| child(&self.root.join("datasets"), name);
        let mut args = Vec::new();
        let (executable, output) = match request {
            Request::Quantize {
                source: name,
                dataset: data,
                validation_dataset,
                algorithm,
                bits,
                group_size,
                context,
                package,
                codec,
            } => {
                if !matches!(bits, 4 | 8) || *group_size == 0 || !group_size.is_multiple_of(32) {
                    return Err(
                        "quantization requires 4/8 bits and a positive multiple-of-32 group size"
                            .into(),
                    );
                }
                args.push(algorithm.command().into());
                append_path(&mut args, "--input", source(name)?);
                append_path(&mut args, "--dataset", dataset(data)?);
                if let Some(data) = validation_dataset {
                    append_path(&mut args, "--validation-dataset", dataset(data)?);
                }
                for (flag, value) in [
                    ("--bits", *bits),
                    ("--group-size", *group_size),
                    ("--max-sequence-length", positive(*context, 8192)?),
                ] {
                    args.extend([flag.into(), value.to_string()]);
                }
                args.extend([
                    "--qif-profile".into(),
                    if *package { "package" } else { "single" }.into(),
                    "--qif-codec".into(),
                    codec.argument().into(),
                ]);
                (&self.converter, Output::Model)
            }
            Request::Calibrate {
                source: name,
                dataset: data,
                context,
            } => {
                args.push("collect-imatrix".into());
                append_path(&mut args, "--input", source(name)?);
                append_path(&mut args, "--dataset", dataset(data)?);
                args.extend([
                    "--max-sequence-length".into(),
                    positive(*context, 8192)?.to_string(),
                ]);
                (&self.cli, Output::Calibration)
            }
            Request::Package {
                source: name,
                codec,
                max_shard_bytes,
            } => {
                let input = source(name)?;
                let gguf = if input.is_file() {
                    let mut magic = [0; 4];
                    fs::File::open(&input)
                        .and_then(|mut file| file.read_exact(&mut magic))
                        .map_err(io_error)?;
                    &magic == b"GGUF"
                } else {
                    false
                };
                args.push(if gguf { "from-gguf" } else { "repack-qif" }.into());
                if gguf {
                    args.extend(["--qif-profile".into(), "package".into()]);
                }
                append_path(&mut args, "--input", input);
                args.extend(["--qif-codec".into(), codec.argument().into()]);
                if let Some(bytes) = max_shard_bytes {
                    args.extend([
                        "--max-shard-bytes".into(),
                        positive(*bytes, usize::MAX)?.to_string(),
                    ]);
                }
                (&self.converter, Output::Package)
            }
            Request::Verify { source: name } => {
                let input = source(name)?;
                let legacy = input.is_dir()
                    && input.join("manifest.json").is_file()
                    && !input.join("qif.json").exists();
                args.push(if legacy { "verify-package" } else { "verify" }.into());
                append_path(&mut args, "--input", input);
                (&self.cli, Output::Verification)
            }
            Request::Benchmark {
                source: name,
                prompt,
                tokens,
                context,
            } => {
                if prompt.trim().is_empty() || prompt.len() > 4096 {
                    return Err("benchmark prompt must contain 1 to 4096 bytes".into());
                }
                args.push("generate".into());
                append_path(&mut args, "--input", source(name)?);
                args.extend([
                    "--prompt".into(),
                    prompt.clone(),
                    "--max-new-tokens".into(),
                    positive(*tokens, 512)?.to_string(),
                    "--max-sequence-length".into(),
                    positive(*context, 8192)?.to_string(),
                ]);
                (&self.cli, Output::Benchmark)
            }
        };
        Ok(Prepared {
            executable: executable.clone(),
            arguments: args,
            output,
        })
    }
}

fn append_path(args: &mut Vec<String>, flag: &str, path: PathBuf) {
    args.push(flag.into());
    args.push(path.to_string_lossy().into_owned());
}

fn positive(value: usize, maximum: usize) -> Result<usize, String> {
    if !(1..=maximum).contains(&value) {
        return Err(format!("expected a value from 1 to {maximum}"));
    }
    Ok(value)
}
