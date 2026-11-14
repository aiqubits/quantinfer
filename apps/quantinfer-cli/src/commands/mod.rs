//! Command dispatch for the QuantInfer CLI.

pub(crate) mod common;
mod generate;
mod package;

use std::env;
use std::path::PathBuf;

use quantinfer::cpu::ModelSource;
use quantinfer_tokenizer::{ChatMessage, Qwen3ChatTemplate, ThinkingMode};

use crate::calibration;

use common::{has_flag, option_value, print_help};
use generate::generate;
use package::{list_profiles, verify_package};

pub(crate) fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    match arguments.first().map(String::as_str) {
        None | Some("--help") | Some("-h") => print_help(),
        Some("--version") => println!("quantinfer {}", env!("CARGO_PKG_VERSION")),
        Some("render-chat") => {
            let options = &arguments[1..];
            let role = option_value(options, "--role")?.unwrap_or_else(|| "user".to_owned());
            let text = option_value(options, "--text")?
                .ok_or_else(|| "render-chat requires --text".to_owned())?;
            let thinking = if has_flag(options, "--think") {
                ThinkingMode::Enabled
            } else {
                ThinkingMode::Disabled
            };
            let prompt = Qwen3ChatTemplate
                .render(&[ChatMessage::new(role, text)], thinking, true)
                .map_err(|error| error.to_string())?;
            print!("{prompt}");
        }
        Some("inspect-gguf") => {
            let path = option_value(&arguments[1..], "--input")?
                .map(PathBuf::from)
                .ok_or_else(|| "inspect-gguf requires --input".to_owned())?;
            let source = ModelSource::open(&path).map_err(|error| error.to_string())?;
            let reader = source
                .open_reader(None)
                .map_err(|error| error.to_string())?;
            if has_flag(&arguments[1..], "--json") {
                println!(
                    "{}",
                    serde_json::json!({
                        "metadata": reader.metadata(),
                        "tensors": reader.tensors(),
                    })
                );
                return Ok(());
            }
            println!("metadata: {}", reader.metadata().len());
            for key in reader.metadata().keys() {
                println!("metadata_key={key}");
            }
            for tensor in reader.tensors() {
                println!(
                    "{} {:?} {:?} {} bytes",
                    tensor.name, tensor.dtype, tensor.shape, tensor.size
                );
            }
        }
        Some("verify-package") => verify_package(&arguments[1..])?,
        Some("list-profiles") => list_profiles(&arguments[1..])?,
        Some("collect-imatrix") => calibration::collect_imatrix(&arguments[1..])?,
        Some("generate") => generate(&arguments[1..])?,
        Some(command) => return Err(format!("unknown command {command}; use --help")),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::common::parse_token_ids;

    #[test]
    fn parses_comma_separated_token_ids() {
        assert_eq!(parse_token_ids("1, 20,3").unwrap(), vec![1, 20, 3]);
        assert!(parse_token_ids("1,nope").is_err());
    }
}
