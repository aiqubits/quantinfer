//! Command dispatch for the QuantInfer CLI.

pub(crate) mod common;
mod generate;
mod inspect;
mod package;

use quantinfer_tokenizer::{ChatMessage, Qwen3ChatTemplate, ThinkingMode};
use std::env;

use crate::calibration;

use common::{has_flag, option_value, print_help, validate_options};
use generate::generate;
use package::{list_profiles, verify_package};

pub(crate) fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    match arguments.first().map(String::as_str) {
        None | Some("--help") | Some("-h") => print_help(),
        Some("--version") => println!("quantinfer {}", env!("CARGO_PKG_VERSION")),
        Some("render-chat") => {
            let options = &arguments[1..];
            validate_options(
                options,
                &[("--role", true), ("--text", true), ("--think", false)],
            )?;
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
            validate_options(&arguments[1..], &[("--input", true), ("--json", false)])?;
            inspect::inspect_gguf(&arguments[1..])?;
        }
        Some("inspect" | "verify") => {
            validate_options(&arguments[1..], &[("--input", true), ("--json", false)])?;
            inspect::native(&arguments[1..], arguments[0] == "verify")?;
        }
        Some("verify-package") => {
            validate_options(&arguments[1..], &[("--input", true), ("--json", false)])?;
            verify_package(&arguments[1..])?
        }
        Some("list-profiles") => {
            validate_options(&arguments[1..], &[("--input", true), ("--json", false)])?;
            list_profiles(&arguments[1..])?
        }
        Some("collect-imatrix") => {
            validate_options(
                &arguments[1..],
                &[
                    ("--input", true),
                    ("--dataset", true),
                    ("--output", true),
                    ("--max-sequence-length", true),
                ],
            )?;
            calibration::collect_imatrix(&arguments[1..])?
        }
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

    #[test]
    fn rejects_unknown_or_incomplete_options() {
        assert!(
            super::common::validate_options(&["--json".to_owned()], &[("--json", false)]).is_ok()
        );
        assert!(
            super::common::validate_options(&["--typo".to_owned()], &[("--json", false)]).is_err()
        );
        assert!(
            super::common::validate_options(&["--input".to_owned()], &[("--input", true)]).is_err()
        );
        assert!(super::common::validate_options(
            &[
                "--input".to_owned(),
                "one.gguf".to_owned(),
                "--input".to_owned(),
                "two.gguf".to_owned(),
            ],
            &[("--input", true)],
        )
        .is_err());
    }
}
