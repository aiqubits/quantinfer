//! Command-line configuration for the HTTP server.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::time::Duration;

use crate::engine;

pub(crate) struct Settings {
    pub(crate) bind: String,
    pub(crate) models: BTreeMap<String, PathBuf>,
    pub(crate) model_registry: Option<PathBuf>,
    pub(crate) context_length: usize,
    pub(crate) maximum_requests: usize,
    pub(crate) batch: engine::BatchLimits,
    pub(crate) kv_budget_bytes: usize,
    pub(crate) request_timeout: Duration,
    pub(crate) requests_per_second: Option<NonZeroUsize>,
}

impl Settings {
    pub(crate) fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut settings = Self {
            bind: "127.0.0.1:8080".to_owned(),
            models: BTreeMap::new(),
            model_registry: None,
            context_length: 512,
            maximum_requests: 8,
            batch: engine::BatchLimits {
                sequences: NonZeroUsize::new(8).unwrap(),
                tokens: NonZeroUsize::new(512).unwrap(),
            },
            kv_budget_bytes: 512 * 1024 * 1024,
            request_timeout: Duration::from_secs(60),
            requests_per_second: None,
        };
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("{flag} requires a value"))?;
            let number = || {
                value
                    .parse::<usize>()
                    .map_err(|error| format!("invalid {flag}: {error}"))
            };
            match flag.as_str() {
                "--bind" => settings.bind = value,
                "--context-length" => settings.context_length = number()?,
                "--max-requests" => settings.maximum_requests = number()?,
                "--max-batch-sequences" | "--max-batch-tokens" => {
                    let limit = NonZeroUsize::new(number()?)
                        .ok_or_else(|| format!("{flag} must be positive"))?;
                    if flag == "--max-batch-sequences" {
                        settings.batch.sequences = limit;
                    } else {
                        settings.batch.tokens = limit;
                    }
                }
                "--kv-budget-bytes" => settings.kv_budget_bytes = number()?,
                "--requests-per-second" => {
                    settings.requests_per_second = Some(
                        NonZeroUsize::new(number()?)
                            .ok_or_else(|| format!("{flag} must be positive"))?,
                    );
                }
                "--request-timeout-ms" => {
                    settings.request_timeout = Duration::from_millis(number()? as u64)
                }
                "--model-registry" => settings.model_registry = Some(PathBuf::from(value)),
                "--model" => {
                    let (name, path) = value
                        .split_once('=')
                        .filter(|(name, path)| !name.is_empty() && !path.is_empty())
                        .ok_or_else(|| "--model requires NAME=PATH".to_owned())?;
                    if settings
                        .models
                        .insert(name.to_owned(), PathBuf::from(path))
                        .is_some()
                    {
                        return Err(format!("duplicate model name {name}"));
                    }
                }
                _ => return Err(format!("unknown server option {flag}")),
            }
        }
        if settings.context_length == 0
            || !(1..=1024).contains(&settings.maximum_requests)
            || settings.kv_budget_bytes == 0
            || settings.request_timeout.is_zero()
        {
            return Err("context, request timeout and KV budget must be positive and --max-requests must be 1..=1024".to_owned());
        }
        Ok(settings)
    }
}
