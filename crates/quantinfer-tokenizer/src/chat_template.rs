use std::io::{self, Write};

use minijinja::{
    context, syntax::SyntaxConfig, value::Serde, Environment, Error, ErrorKind, Value,
};
use serde::Serialize;
use serde_json::Value as JsonValue;

use crate::{ChatMessage, ChatTemplate, ChatTemplateOptions, Result, ThinkingMode, TokenizerError};

const TEMPLATE_NAME: &str = "chat";

/// A Jinja chat template loaded from a Hugging Face tokenizer configuration.
#[derive(Debug, Clone)]
pub struct HuggingFaceChatTemplate {
    environment: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

impl HuggingFaceChatTemplate {
    /// Compile the `chat_template` string in a `tokenizer_config.json` document.
    pub fn from_config_json(config: &str) -> Result<Self> {
        let config: JsonValue = serde_json::from_str(config).map_err(|error| {
            TokenizerError::InvalidInput(format!("invalid tokenizer configuration: {error}"))
        })?;
        let source = config["chat_template"].as_str().ok_or_else(|| {
            TokenizerError::InvalidInput(
                "tokenizer configuration has no chat_template string".to_owned(),
            )
        })?;
        let mut environment = Environment::new();
        let syntax = SyntaxConfig::builder()
            .trim_blocks(true)
            .lstrip_blocks(true)
            .build()
            .map_err(|error| {
                TokenizerError::InvalidInput(format!("invalid chat template syntax: {error}"))
            })?;
        environment.set_syntax(syntax);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_filter("tojson", python_style_json);
        environment.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        environment
            .add_template_owned(TEMPLATE_NAME, source.to_owned())
            .map_err(render_error)?;
        let special_token = |name: &str| {
            let value = &config[name];
            value
                .as_str()
                .or_else(|| value["content"].as_str())
                .unwrap_or_default()
                .to_owned()
        };
        Ok(Self {
            environment,
            bos_token: special_token("bos_token"),
            eos_token: special_token("eos_token"),
        })
    }

    /// Render JSON messages with optional tool definitions and structured calls.
    pub fn render_structured(
        &self,
        messages: &[JsonValue],
        tools: &[JsonValue],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        if messages.is_empty() {
            return Err(TokenizerError::InvalidInput(
                "chat template requires at least one message".to_owned(),
            ));
        }
        self.environment
            .get_template(TEMPLATE_NAME)
            .and_then(|template| {
                template.render(context! {
                    messages => Serde(messages),
                    tools => Serde(tools),
                    bos_token => self.bos_token.as_str(),
                    eos_token => self.eos_token.as_str(),
                    add_bos_token => options.add_bos_token,
                    add_generation_prompt => options.add_generation_prompt,
                    enable_thinking => options.thinking == ThinkingMode::Enabled,
                })
            })
            .map_err(render_error)
    }
}

impl ChatTemplate for HuggingFaceChatTemplate {
    fn render_messages(
        &self,
        messages: &[ChatMessage],
        options: ChatTemplateOptions,
    ) -> Result<String> {
        let messages = messages
            .iter()
            .map(|message| serde_json::json!({"role": message.role, "content": message.content}))
            .collect::<Vec<_>>();
        self.render_structured(&messages, &[], options)
    }
}

fn render_error(error: Error) -> TokenizerError {
    TokenizerError::InvalidInput(format!("chat template error: {error}"))
}

// Transformers uses json.dumps with a space after each separator by default.
struct PythonJsonFormatter;

impl serde_json::ser::Formatter for PythonJsonFormatter {
    fn begin_array_value<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.begin_array_value(writer, first)
    }

    fn begin_object_value<W: Write + ?Sized>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

fn python_style_json(value: Value) -> std::result::Result<String, Error> {
    let mut bytes = Vec::new();
    value
        .serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes,
            PythonJsonFormatter,
        ))
        .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))?;
    String::from_utf8(bytes)
        .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))
}
