//! Command dispatch for the QuantInfer CLI.

pub(crate) mod common;
mod generate;
mod inspect;
mod package;

use quantinfer_tokenizer::{ChatMessage, Qwen3ChatTemplate, ThinkingMode};
use std::env;

use crate::calibration;

use common::{parse_options, print_help};
use generate::generate;
use package::{list_profiles, verify_package};

pub(crate) fn run() -> Result<(), String> {
    let arguments: Vec<String> = env::args().skip(1).collect();
    match arguments.first().map(String::as_str) {
        None | Some("--help") | Some("-h") => print_help(),
        Some("--version") => println!("quantinfer {}", env!("CARGO_PKG_VERSION")),
        Some("render-chat") => {
            let options = parse_options(
                &arguments[1..],
                &[("--role", true), ("--text", true), ("--think", false)],
            )?;
            let role = options.value("--role").unwrap_or("user");
            let text = options
                .value("--text")
                .ok_or_else(|| "render-chat requires --text".to_owned())?;
            let thinking = if options.has("--think") {
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
            let options = parse_options(&arguments[1..], &[("--input", true), ("--json", false)])?;
            inspect::inspect_gguf(&options)?;
        }
        Some("inspect" | "verify") => {
            let options = parse_options(&arguments[1..], &[("--input", true), ("--json", false)])?;
            inspect::native(&options, arguments[0] == "verify")?;
        }
        Some("verify-package") => {
            let options = parse_options(&arguments[1..], &[("--input", true), ("--json", false)])?;
            verify_package(&options)?
        }
        Some("list-profiles") => {
            let options = parse_options(&arguments[1..], &[("--input", true), ("--json", false)])?;
            list_profiles(&options)?
        }
        Some("collect-imatrix") => {
            let options = parse_options(
                &arguments[1..],
                &[
                    ("--input", true),
                    ("--dataset", true),
                    ("--output", true),
                    ("--max-sequence-length", true),
                ],
            )?;
            calibration::collect_imatrix(&options)?
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
        assert!(super::common::parse_options(&["--json".to_owned()], &[("--json", false)]).is_ok());
        assert!(
            super::common::parse_options(&["--typo".to_owned()], &[("--json", false)]).is_err()
        );
        assert!(
            super::common::parse_options(&["--input".to_owned()], &[("--input", true)]).is_err()
        );
        assert!(super::common::parse_options(
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

    #[test]
    fn option_values_can_match_other_flag_names() {
        let arguments = [
            "--prompt",
            "--profile-report",
            "--profile-report",
            "report.json",
            "--role",
            "--chat",
        ]
        .map(str::to_owned);
        let options = super::common::parse_options(
            &arguments,
            &[
                ("--prompt", true),
                ("--profile-report", true),
                ("--role", true),
                ("--chat", false),
            ],
        )
        .unwrap();
        assert_eq!(options.value("--prompt"), Some("--profile-report"));
        assert_eq!(options.value("--profile-report"), Some("report.json"));
        assert_eq!(options.value("--role"), Some("--chat"));
        assert!(!options.has("--chat"));
    }
}
