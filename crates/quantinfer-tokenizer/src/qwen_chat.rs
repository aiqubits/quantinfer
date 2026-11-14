//! Native Qwen3 chat template.

use super::{ChatMessage, ChatTemplate, ChatTemplateOptions, Result, ThinkingMode, TokenizerError};

#[derive(Debug, Clone, Copy, Default)]
pub struct Qwen3ChatTemplate;

impl Qwen3ChatTemplate {
    pub fn render(
        &self,
        messages: &[ChatMessage],
        thinking: ThinkingMode,
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.render_messages(
            messages,
            ChatTemplateOptions {
                thinking,
                add_generation_prompt,
                add_bos_token: false,
            },
        )
    }
}

impl ChatTemplate for Qwen3ChatTemplate {
    fn render_messages(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(TokenizerError::InvalidInput(
                "chat template requires at least one message".to_owned(),
            ));
        }
        let mut output = String::new();
        if options.add_bos_token {
            output.push_str("<|im_start|>");
        }
        for message in messages {
            if message.role.is_empty() {
                return Err(TokenizerError::InvalidInput(
                    "chat message role cannot be empty".to_owned(),
                ));
            }
            output.push_str("<|im_start|>");
            output.push_str(&message.role);
            output.push('\n');
            output.push_str(&message.content);
            output.push_str("<|im_end|>\n");
        }
        if options.add_generation_prompt {
            output.push_str("<|im_start|>assistant\n");
            if options.thinking == ThinkingMode::Disabled {
                output.push_str("<think>\n\n</think>\n\n");
            }
        }
        Ok(output)
    }
}
