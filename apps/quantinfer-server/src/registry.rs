use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use quantinfer::cpu::{
    estimated_weight_bytes, load_gguf_tokenizer, resolve_profiled_backend, CpuModel, ModelSource,
};
use quantinfer::format::{ExecutionVariant, GgufReader, ModelManifest};
use quantinfer::runtime::{InferenceProfile, MemoryPlan};
use quantinfer::tokenizer::GgufBpeTokenizer;
use serde::{Deserialize, Serialize};

pub(crate) const CPU_SCALAR_PROVIDER: &str = "cpu-scalar";

#[derive(Clone, Deserialize, Serialize, PartialEq)]
pub(crate) struct ModelEntry {
    pub(crate) path: PathBuf,
    pub(crate) revision: Option<String>,
    pub(crate) lora: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) profile: Option<InferenceProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) profile_artifact_sha256: Option<String>,
}

pub struct ServedModel {
    entry: ModelEntry,
    managed: bool,
    pub decoder: Arc<CpuModel>,
    pub tokenizer: Option<GgufBpeTokenizer>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelDescriptor {
    pub id: String,
    pub artifact_name: String,
    pub revision: Option<String>,
    pub has_lora: bool,
    pub package: bool,
    pub profiles: Vec<String>,
    pub managed: bool,
    pub applied_profile: Option<InferenceProfile>,
}

impl ServedModel {
    fn load(entry: ModelEntry, managed: bool) -> Result<Self, String> {
        if let Some(expected) = &entry.profile_artifact_sha256 {
            if &artifact_sha256(&entry.path)? != expected {
                return Err("stored inference profile is bound to a different artifact".into());
            }
        }
        let budget = entry
            .profile
            .as_ref()
            .and_then(|profile| profile.memory_budget_bytes);
        if budget.is_some() && entry.lora.is_some() {
            return Err("LoRA with a memory budget is unsupported".into());
        }
        let source = ModelSource::open(&entry.path).map_err(|error| error.to_string())?;
        let reader = source
            .open_reader(budget)
            .map_err(|error| error.to_string())?;
        if managed && reader.qif_metadata().is_some() {
            reader.verify_qif().map_err(|error| error.to_string())?;
        }
        let tokenizer = reader
            .metadata_value("tokenizer.ggml.tokens")
            .map_err(|error| error.to_string())?
            .is_some()
            .then(|| load_gguf_tokenizer(&reader))
            .transpose()
            .map_err(|error| error.to_string())?;
        let manifest = source.package().map(|package| &package.manifest);
        let profiled = entry
            .profile
            .as_ref()
            .map(|profile| validated_profile(&reader, manifest, profile))
            .transpose()?;
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
        let mut decoder = CpuModel::from_gguf(
            &reader,
            profiled.as_ref().or(manifest),
            entry.profile.as_ref(),
        )
        .map_err(|error| error.to_string())?;
        if let Some(adapter) = adapter {
            decoder
                .attach_lora(adapter)
                .map_err(|error| error.to_string())?;
        }
        Ok(Self {
            entry,
            managed,
            decoder: Arc::new(decoder),
            tokenizer,
        })
    }

    pub(crate) fn context_length(&self) -> usize {
        self.entry.profile.as_ref().map_or_else(
            || self.decoder.max_position_embeddings(),
            |profile| profile.context_length,
        )
    }
}

pub(crate) fn artifact_sha256(path: &Path) -> Result<String, String> {
    let manifest = path.join("qif.json");
    let target = if manifest.is_file() {
        manifest
    } else if path.is_dir() {
        path.join("manifest.json")
    } else {
        path.to_owned()
    };
    let mut file = fs::File::open(target).map_err(|error| error.to_string())?;
    quantinfer::format::sha256_reader(&mut file).map_err(|error| error.to_string())
}

pub(crate) fn validated_profile(
    reader: &GgufReader,
    manifest: Option<&ModelManifest>,
    profile: &InferenceProfile,
) -> Result<ModelManifest, String> {
    let mut manifest = manifest
        .or(reader.embedded_manifest())
        .cloned()
        .ok_or("inference profiles require a model manifest")?;
    if manifest.execution_variants.is_empty() {
        manifest.execution_variants.push(ExecutionVariant {
            name: profile.model_variant.clone(),
            quantization: manifest.quantization,
            dtype: "f32-accumulation".into(),
            runtime_provider: CPU_SCALAR_PROVIDER.into(),
            kernel_provider: CPU_SCALAR_PROVIDER.into(),
        });
    }
    let spec =
        CpuModel::spec_from_gguf(reader, Some(&manifest)).map_err(|error| error.to_string())?;
    if profile.context_length > spec.max_sequence_length {
        return Err(format!(
            "inference profile context exceeds model limit {}",
            spec.max_sequence_length
        ));
    }
    resolve_profiled_backend(reader, &manifest, profile).map_err(|error| error.to_string())?;
    if let Some(budget) = profile.memory_budget_bytes {
        let weights = estimated_weight_bytes(reader, &spec).map_err(|error| error.to_string())?;
        let largest_tensor = reader
            .tensors()
            .iter()
            .map(|tensor| tensor.size)
            .max()
            .unwrap_or(0);
        let loading = reader
            .loading_memory_bytes()
            .map_err(|error| error.to_string())?
            .checked_add(weights)
            .and_then(|bytes| bytes.checked_add(largest_tensor))
            .ok_or("model memory estimate exceeds usize")?;
        if loading > budget {
            return Err(format!(
                "profile budget {budget} is below the model loading estimate {loading}"
            ));
        }
        MemoryPlan::for_dense_prefill(
            &spec,
            weights,
            profile.context_length,
            profile.context_length,
        )
        .and_then(|plan| plan.ensure_budget(budget))
        .map_err(|error| error.to_string())?;
    }
    Ok(manifest)
}

pub struct ModelRegistry {
    fixed: BTreeMap<String, PathBuf>,
    config: Option<PathBuf>,
    managed: Option<PathBuf>,
    changes: Mutex<()>,
    models: RwLock<BTreeMap<String, Arc<ServedModel>>>,
}

impl ModelRegistry {
    pub fn new(
        fixed: BTreeMap<String, PathBuf>,
        config: Option<PathBuf>,
        managed: Option<PathBuf>,
    ) -> Result<Self, String> {
        let registry = Self {
            fixed,
            config,
            managed,
            changes: Mutex::default(),
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
                    artifact_name: model
                        .entry
                        .path
                        .file_name()
                        .map(|name| name.to_string_lossy().into_owned())
                        .unwrap_or_else(|| "model".to_owned()),
                    revision: model.entry.revision.clone(),
                    has_lora: model.entry.lora.is_some(),
                    package: model.entry.path.is_dir(),
                    profiles,
                    managed: model.managed,
                    applied_profile: model.entry.profile.clone(),
                }
            })
            .collect()
    }

    fn refresh(&self) -> Result<(), String> {
        let _change = self
            .changes
            .lock()
            .map_err(|_| "model registry lock failed")?;
        let replacement = self.prepare(&self.managed_entries()?, None)?;
        self.activate(replacement);
        Ok(())
    }

    pub(crate) fn change_managed(
        &self,
        name: String,
        entry: Option<ModelEntry>,
    ) -> Result<(), String> {
        let _change = self
            .changes
            .lock()
            .map_err(|_| "model registry lock failed")?;
        let path = self
            .managed
            .as_deref()
            .ok_or("Studio registry management is disabled")?;
        let mut managed = self.managed_entries()?;
        if let Some(entry) = entry {
            managed.insert(name.clone(), entry);
        } else if managed.remove(&name).is_none() {
            return Err(format!("no Studio model named {name}"));
        }
        let replacement = self.prepare(&managed, Some(&name))?;
        let parent = path
            .parent()
            .ok_or("Studio registry has no parent directory")?;
        let mut stage = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
        serde_json::to_writer(&mut stage, &managed).map_err(|e| e.to_string())?;
        stage
            .flush()
            .and_then(|()| stage.as_file().sync_all())
            .map_err(|e| e.to_string())?;
        stage.persist(path).map_err(|e| e.to_string())?;
        self.activate(replacement);
        Ok(())
    }

    fn managed_entries(&self) -> Result<BTreeMap<String, ModelEntry>, String> {
        match &self.managed {
            Some(path) if path.exists() => read_entries(path),
            _ => Ok(BTreeMap::new()),
        }
    }

    fn prepare(
        &self,
        managed: &BTreeMap<String, ModelEntry>,
        reload: Option<&str>,
    ) -> Result<BTreeMap<String, Arc<ServedModel>>, String> {
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
                        profile: None,
                        profile_artifact_sha256: None,
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
        for (name, entry) in managed {
            if desired.insert(name.clone(), entry.clone()).is_some() {
                return Err(format!(
                    "Studio model {name} conflicts with the server configuration"
                ));
            }
        }
        let current = self
            .models
            .read()
            .map_err(|_| "model registry lock failed")?
            .clone();
        let mut replacement = BTreeMap::new();
        for (name, entry) in desired {
            let is_managed = managed.contains_key(&name);
            let model = if let Some(existing) = current.get(&name).filter(|model| {
                reload != Some(&name) && model.entry == entry && model.managed == is_managed
            }) {
                existing.clone()
            } else {
                Arc::new(
                    ServedModel::load(entry, is_managed)
                        .map_err(|error| format!("cannot load {name}: {error}"))?,
                )
            };
            replacement.insert(name, model);
        }
        Ok(replacement)
    }

    fn activate(&self, replacement: BTreeMap<String, Arc<ServedModel>>) {
        // Requests retain their model Arc while a fully loaded replacement becomes visible.
        *self
            .models
            .write()
            .unwrap_or_else(|error| error.into_inner()) = replacement;
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
