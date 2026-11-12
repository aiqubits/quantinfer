use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use quantinfer::cpu::{load_gguf_tokenizer, CpuModel, ModelSource};
use quantinfer::tokenizer::GgufBpeTokenizer;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, PartialEq, Eq)]
struct ModelEntry {
    path: PathBuf,
    revision: Option<String>,
    lora: Option<PathBuf>,
}

pub struct ServedModel {
    entry: ModelEntry,
    pub decoder: Arc<CpuModel>,
    pub tokenizer: Option<GgufBpeTokenizer>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub path: String,
    pub revision: Option<String>,
    pub has_lora: bool,
    pub package: bool,
    pub profiles: Vec<String>,
}

impl ServedModel {
    fn load(entry: ModelEntry) -> Result<Self, String> {
        let source = ModelSource::open(&entry.path).map_err(|error| error.to_string())?;
        let reader = source
            .open_reader(None)
            .map_err(|error| error.to_string())?;
        let tokenizer = reader
            .metadata()
            .contains_key("tokenizer.ggml.tokens")
            .then(|| load_gguf_tokenizer(&reader))
            .transpose()
            .map_err(|error| error.to_string())?;
        let manifest = source.package().map(|package| &package.manifest);
        let adapter = entry
            .lora
            .as_ref()
            .map(|path| {
                let spec = CpuModel::spec_from_gguf(&reader, manifest)
                    .map_err(|error| error.to_string())?;
                quantinfer_adapter_lora::load_peft(path, &spec).map_err(|error| {
                    format!("cannot load LoRA adapter {}: {error}", path.display())
                })
            })
            .transpose()?;
        let mut decoder =
            CpuModel::from_gguf(&reader, manifest, None).map_err(|error| error.to_string())?;
        if let Some(adapter) = adapter {
            decoder
                .attach_lora(adapter)
                .map_err(|error| error.to_string())?;
        }
        Ok(Self {
            entry,
            decoder: Arc::new(decoder),
            tokenizer,
        })
    }
}

pub struct ModelRegistry {
    fixed: BTreeMap<String, PathBuf>,
    config: Option<PathBuf>,
    models: RwLock<BTreeMap<String, Arc<ServedModel>>>,
}

impl ModelRegistry {
    pub fn new(fixed: BTreeMap<String, PathBuf>, config: Option<PathBuf>) -> Result<Self, String> {
        let registry = Self {
            fixed,
            config,
            models: RwLock::default(),
        };
        registry.refresh()?;
        Ok(registry)
    }

    pub fn get(&self, name: &str) -> Option<Arc<ServedModel>> {
        self.models
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .get(name)
            .cloned()
    }

    pub fn names(&self) -> Vec<String> {
        self.models
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .keys()
            .cloned()
            .collect()
    }

    pub fn descriptors(&self) -> Vec<ModelDescriptor> {
        self.models
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .map(|(id, model)| {
                let profiles = model
                    .entry
                    .path
                    .is_dir()
                    .then(|| model.entry.path.join("profiles"))
                    .and_then(|path| fs::read_dir(path).ok())
                    .into_iter()
                    .flatten()
                    .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                    .filter(|path| {
                        path.extension()
                            .is_some_and(|extension| extension == "json")
                    })
                    .filter_map(|path| path.file_stem().map(|name| name.to_string_lossy().into()))
                    .collect();
                ModelDescriptor {
                    id: id.clone(),
                    path: model.entry.path.display().to_string(),
                    revision: model.entry.revision.clone(),
                    has_lora: model.entry.lora.is_some(),
                    package: model.entry.path.is_dir(),
                    profiles,
                }
            })
            .collect()
    }

    fn refresh(&self) -> Result<(), String> {
        let mut desired: BTreeMap<_, _> = self
            .fixed
            .iter()
            .map(|(name, path)| {
                (
                    name.clone(),
                    ModelEntry {
                        path: path.clone(),
                        revision: None,
                        lora: None,
                    },
                )
            })
            .collect();
        if let Some(path) = &self.config {
            let entries = read_entries(path)?;
            for (name, entry) in entries {
                if desired.insert(name.clone(), entry).is_some() {
                    return Err(format!("registry model {name} conflicts with --model"));
                }
            }
        }
        let current = self
            .models
            .read()
            .map_err(|_| "model registry lock failed")?
            .clone();
        let mut replacement = BTreeMap::new();
        for (name, entry) in desired {
            let model =
                if let Some(existing) = current.get(&name).filter(|model| model.entry == entry) {
                    existing.clone()
                } else {
                    Arc::new(
                        ServedModel::load(entry)
                            .map_err(|error| format!("cannot load {name}: {error}"))?,
                    )
                };
            replacement.insert(name, model);
        }
        // Requests retain their model Arc while a fully loaded replacement becomes visible.
        *self
            .models
            .write()
            .map_err(|_| "model registry lock failed")? = replacement;
        Ok(())
    }

    #[cfg(unix)]
    pub fn watch(self: &Arc<Self>) -> Result<Option<tokio::task::JoinHandle<()>>, String> {
        if self.config.is_none() {
            return Ok(None);
        }
        let mut signal = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
            .map_err(|error| format!("cannot watch model registry: {error}"))?;
        let registry = self.clone();
        Ok(Some(tokio::spawn(async move {
            while signal.recv().await.is_some() {
                let registry = registry.clone();
                match tokio::task::spawn_blocking(move || registry.refresh()).await {
                    Ok(Ok(())) => eprintln!("quantinfer-server: model registry reloaded"),
                    Ok(Err(error)) => {
                        eprintln!("quantinfer-server: model registry reload failed: {error}")
                    }
                    Err(error) => {
                        eprintln!("quantinfer-server: model registry worker failed: {error}")
                    }
                }
            }
        })))
    }
}

fn read_entries(path: &Path) -> Result<BTreeMap<String, ModelEntry>, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("cannot read registry {}: {error}", path.display()))?;
    let mut entries: BTreeMap<String, ModelEntry> = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse registry {}: {error}", path.display()))?;
    let base = path.parent().unwrap_or_else(|| Path::new("."));
    for (name, entry) in &mut entries {
        if name.is_empty() || entry.path.as_os_str().is_empty() {
            return Err("model registry entries require a name and path".to_owned());
        }
        entry.path = base.join(&entry.path);
        if let Some(adapter) = &mut entry.lora {
            if adapter.as_os_str().is_empty() {
                return Err(format!(
                    "model registry entry {name} has an empty LoRA path"
                ));
            }
            *adapter = base.join(adapter.as_path());
        }
    }
    Ok(entries)
}
