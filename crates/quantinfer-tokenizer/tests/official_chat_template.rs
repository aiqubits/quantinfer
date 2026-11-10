use std::{env, fs};

use quantinfer_tokenizer::{ChatTemplateOptions, HuggingFaceChatTemplate, ThinkingMode};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    messages: Vec<Value>,
    tools: Vec<Value>,
    enable_thinking: bool,
    add_generation_prompt: bool,
    rendered: String,
}

#[test]
#[ignore = "requires the pinned official Qwen3 tokenizer_config.json"]
fn renders_official_qwen3_chat_template_golden_cases() {
    let path = env::var_os("QUANTINFER_TOKENIZER_CONFIG")
        .expect("set QUANTINFER_TOKENIZER_CONFIG to the verified tokenizer_config.json");
    let config = fs::read_to_string(path).unwrap();
    let template = HuggingFaceChatTemplate::from_config_json(&config).unwrap();
    let fixture: Fixture = serde_json::from_str(include_str!(
        "../../../tests/golden/qwen3-chat-template.json"
    ))
    .unwrap();
    for case in fixture.cases {
        let rendered = template
            .render_structured(
                &case.messages,
                &case.tools,
                ChatTemplateOptions {
                    thinking: if case.enable_thinking {
                        ThinkingMode::Enabled
                    } else {
                        ThinkingMode::Disabled
                    },
                    add_generation_prompt: case.add_generation_prompt,
                    add_bos_token: false,
                },
            )
            .unwrap_or_else(|error| panic!("{}: {error}", case.name));
        assert_eq!(rendered, case.rendered, "{}", case.name);
    }
}
