//! Typed local jobs backed by the same executables used from the command line.

mod output;
mod request;

pub(super) use output::read_json;
use request::Prepared;
pub(crate) use request::Request;

use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::{NamedTempFile, TempDir};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::sync::{oneshot, Notify, Semaphore};

const MAX_ACTIVE: usize = 8;
const MAX_HISTORY: usize = 128;
const MAX_LOG: usize = 32 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    Queued,
    Running,
    Succeeded,
    Failed,
    Cancelled,
}

impl Phase {
    fn active(self) -> bool {
        matches!(self, Self::Queued | Self::Running)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Job {
    id: String,
    request: Request,
    phase: Phase,
    result: Option<Value>,
    detail: Option<String>,
}

struct Entry {
    job: Job,
    cancel: Option<oneshot::Sender<()>>,
}

#[derive(Default)]
struct History {
    entries: BTreeMap<String, Entry>,
    closing: bool,
}

pub(crate) struct Jobs {
    root: PathBuf,
    converter: PathBuf,
    cli: PathBuf,
    address: SocketAddr,
    history: Mutex<History>,
    running: Semaphore,
    changed: Notify,
}

#[derive(Serialize)]
pub(crate) struct Inventory {
    sources: Vec<String>,
    datasets: Vec<String>,
    adapters: Vec<String>,
    profiles: Vec<String>,
}

impl Jobs {
    pub(crate) fn new(
        root: &Path,
        converter: &Path,
        cli: &Path,
        address: SocketAddr,
    ) -> Result<Arc<Self>, String> {
        if !address.ip().is_loopback() {
            return Err("Studio management requires a loopback listen address".into());
        }
        fs::create_dir_all(root).map_err(io_error)?;
        let root = root.canonicalize().map_err(io_error)?;
        for name in ["sources", "datasets", "artifacts", "adapters", "profiles"] {
            fs::create_dir_all(root.join(name)).map_err(io_error)?;
            if child(&root, name)? != root.join(name) {
                return Err(format!("Studio directory {name} must be a local directory"));
            }
        }
        let executable = |path: &Path| {
            let path = path.canonicalize().map_err(io_error)?;
            if !path.is_file() {
                return Err("Studio executable path must identify a file".to_owned());
            }
            Ok(path)
        };
        let history = History::load(&root.join("artifacts"))?;
        Ok(Arc::new(Self {
            root,
            converter: executable(converter)?,
            cli: executable(cli)?,
            address,
            history: Mutex::new(history),
            running: Semaphore::new(1),
            changed: Notify::new(),
        }))
    }

    pub(crate) fn accepts_host(&self, host: &str) -> bool {
        let Ok(authority) = host.parse::<axum::http::uri::Authority>() else {
            return false;
        };
        let correct_port = authority.port_u16().unwrap_or(80) == self.address.port();
        let host = authority.host().trim_matches(['[', ']']);
        correct_port
            && (host.eq_ignore_ascii_case("localhost")
                || host.parse::<std::net::IpAddr>().ok() == Some(self.address.ip()))
    }

    pub(crate) fn list(&self) -> Vec<Job> {
        self.history
            .lock()
            .expect("job history lock")
            .entries
            .values()
            .rev()
            .map(|entry| entry.job.clone())
            .collect()
    }

    pub(crate) fn inventory(&self) -> Result<Inventory, String> {
        let mut sources = list_names(&self.root.join("sources"))?;
        for id in list_names(&self.root.join("artifacts"))? {
            let name = format!("job:{id}");
            if self.source(&name).is_ok() {
                sources.push(name);
            }
        }
        let datasets = list_names(&self.root.join("datasets"))?
            .into_iter()
            .filter(|name| name.ends_with(".jsonl"))
            .collect();
        let adapters = list_names(&self.root.join("adapters"))?;
        let profiles = list_names(&self.profile_directory())?
            .into_iter()
            .filter_map(|name| name.strip_suffix(".json").map(str::to_owned))
            .collect();
        Ok(Inventory {
            sources,
            datasets,
            adapters,
            profiles,
        })
    }

    pub(crate) fn save_dataset(&self, name: &str, text: &str) -> Result<(), String> {
        validate_name(name)?;
        if !name.ends_with(".jsonl") {
            return Err("dataset name must end in .jsonl".into());
        }
        let mut count = 0;
        for (index, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let value: Value = serde_json::from_str(line)
                .map_err(|error| format!("dataset line {}: {error}", index + 1))?;
            let valid = value
                .as_str()
                .or_else(|| value.get("text").and_then(Value::as_str));
            if valid.is_none_or(|prompt| prompt.trim().is_empty()) {
                return Err(format!(
                    "dataset line {} requires a nonempty prompt",
                    index + 1
                ));
            }
            count += 1;
        }
        if count == 0 {
            return Err("dataset requires at least one prompt".into());
        }
        let datasets = self.root.join("datasets");
        let mut stage = NamedTempFile::new_in(&datasets).map_err(io_error)?;
        stage.write_all(text.as_bytes()).map_err(io_error)?;
        stage.as_file().sync_all().map_err(io_error)?;
        stage.persist(datasets.join(name)).map_err(io_error)?;
        Ok(())
    }

    pub(crate) fn registry_path(&self) -> PathBuf {
        self.root.join("registry.json")
    }

    pub(super) fn profile_directory(&self) -> PathBuf {
        self.root.join("profiles")
    }

    pub(super) fn adapter(&self, name: &str) -> Result<PathBuf, String> {
        let directory = child(&self.root.join("adapters"), name)?;
        if !directory.is_dir() {
            return Err("LoRA adapter must be a directory".into());
        }
        Ok(directory)
    }

    pub(super) fn source(&self, name: &str) -> Result<PathBuf, String> {
        if let Some(id) = name.strip_prefix("job:") {
            if id.is_empty() || !id.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err("invalid Studio job identifier".into());
            }
            let artifact = child(&self.root.join("artifacts"), id)?;
            return child(&artifact, "model.qif");
        }
        child(&self.root.join("sources"), name)
    }

    pub(crate) fn submit(self: &Arc<Self>, request: Request) -> Result<Job, String> {
        let prepared = self.prepare(&request)?;
        let (cancel, receiver) = oneshot::channel();
        let mut history = self.history.lock().expect("job history lock");
        if history.closing {
            return Err("Studio is shutting down".into());
        }
        let entries = &mut history.entries;
        if entries
            .values()
            .filter(|entry| entry.job.phase.active())
            .count()
            >= MAX_ACTIVE
        {
            return Err("Studio job queue is full".into());
        }
        if entries.len() >= MAX_HISTORY {
            let oldest = entries
                .iter()
                .find(|(_, entry)| !entry.job.phase.active())
                .map(|(id, _)| id.clone())
                .expect("the queue bound reserves room for finished jobs");
            entries.remove(&oldest);
        }
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io_error)?
            .as_nanos()
            .to_string();
        if entries.contains_key(&id) || self.root.join("artifacts").join(&id).exists() {
            return Err("Studio job identifier already exists".into());
        }
        let job = Job {
            id: id.clone(),
            request,
            phase: Phase::Queued,
            result: None,
            detail: None,
        };
        entries.insert(
            id.clone(),
            Entry {
                job: job.clone(),
                cancel: Some(cancel),
            },
        );
        drop(history);
        let jobs = self.clone();
        tokio::spawn(async move {
            jobs.run(id, prepared, receiver).await;
        });
        Ok(job)
    }

    pub(crate) fn cancel(&self, id: &str) -> bool {
        let mut history = self.history.lock().expect("job history lock");
        let Some(entry) = history.entries.get_mut(id) else {
            return false;
        };
        entry
            .cancel
            .take()
            .is_some_and(|cancel| cancel.send(()).is_ok())
    }

    pub(crate) async fn close(&self) {
        {
            let mut history = self.history.lock().expect("job history lock");
            history.closing = true;
            for entry in history.entries.values_mut() {
                if let Some(cancel) = entry.cancel.take() {
                    let _ = cancel.send(());
                }
            }
        }
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if self.list().iter().all(|job| !job.phase.active()) {
                return;
            }
            changed.await;
        }
    }

    async fn run(&self, id: String, prepared: Prepared, mut cancel: oneshot::Receiver<()>) {
        let permit = tokio::select! {
            biased;
            _ = &mut cancel => None,
            permit = self.running.acquire() => permit.ok(),
        };
        let outcome = if let Some(_permit) = permit {
            self.update(&id, Phase::Running, None, None);
            self.execute(&id, prepared, &mut cancel).await
        } else {
            Outcome::Cancelled
        };
        match outcome {
            Outcome::Succeeded(result) => self.update(&id, Phase::Succeeded, Some(result), None),
            Outcome::Failed(error) => self.update(&id, Phase::Failed, None, Some(error)),
            Outcome::Cancelled => self.update(&id, Phase::Cancelled, None, None),
        }
    }

    fn update(&self, id: &str, phase: Phase, result: Option<Value>, detail: Option<String>) {
        let mut history = self.history.lock().expect("job history lock");
        let entry = history.entries.get_mut(id).expect("active Studio job");
        entry.job.phase = phase;
        entry.job.result = result;
        entry.job.detail = detail;
        if !phase.active() {
            entry.cancel = None;
        }
        self.changed.notify_waiters();
    }

    async fn execute(
        &self,
        id: &str,
        prepared: Prepared,
        cancel: &mut oneshot::Receiver<()>,
    ) -> Outcome {
        let artifacts = self.root.join("artifacts");
        let stage = match TempDir::new_in(&artifacts) {
            Ok(stage) => stage,
            Err(error) => return Outcome::Failed(io_error(error)),
        };
        let mut cmd = Command::new(&prepared.executable);
        cmd.args(prepared.arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (flag, name) in prepared.output.arguments() {
            cmd.arg(flag).arg(stage.path().join(name));
        }
        let mut child = match cmd.spawn() {
            Ok(child) => child,
            Err(error) => return Outcome::Failed(io_error(error)),
        };
        let err = tokio::spawn(log_tail(child.stderr.take().expect("piped stderr")));
        let status = tokio::select! {
            biased;
            _ = &mut *cancel => {
                let _ = child.kill().await;
                let _ = child.wait().await;
                err.abort();
                return Outcome::Cancelled;
            }
            status = child.wait() => status,
        };
        let stderr = err.await.unwrap_or_default();
        match status {
            Err(error) => return Outcome::Failed(io_error(error)),
            Ok(status) if !status.success() => {
                return Outcome::Failed(format!("{status}\n{stderr}"));
            }
            Ok(_) => {}
        }
        let check_path = stage.path().to_owned();
        let checked =
            tokio::task::spawn_blocking(move || prepared.output.verify(&check_path)).await;
        let result = match checked {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => return Outcome::Failed(error),
            Err(error) => return Outcome::Failed(io_error(error)),
        };
        let mut history = self.history.lock().expect("job history lock");
        let entry = history.entries.get_mut(id).expect("active Studio job");
        if entry.cancel.take().is_none() {
            return Outcome::Cancelled;
        }
        let mut record = entry.job.clone();
        record.phase = Phase::Succeeded;
        record.result = Some(result.clone());
        drop(history);
        let data = match serde_json::to_vec(&record) {
            Ok(data) => data,
            Err(error) => return Outcome::Failed(io_error(error)),
        };
        if let Err(error) = fs::write(stage.path().join("result.json"), data)
            .and_then(|()| fs::rename(stage.path(), artifacts.join(id)))
        {
            return Outcome::Failed(io_error(error));
        }
        Outcome::Succeeded(result)
    }
}

impl History {
    fn load(artifacts: &Path) -> Result<Self, String> {
        let mut history = Self::default();
        for id in list_names(artifacts)?.into_iter().rev().take(MAX_HISTORY) {
            if !id.bytes().all(|byte| byte.is_ascii_digit()) {
                continue;
            }
            let Ok(root) = child(artifacts, &id) else {
                continue;
            };
            let Ok(record) = read_json(&root.join("result.json"), 64 * 1024) else {
                continue;
            };
            let Ok(job) = serde_json::from_value::<Job>(record) else {
                continue;
            };
            if job.id == id && job.phase == Phase::Succeeded {
                history.entries.insert(id, Entry { job, cancel: None });
            }
        }
        Ok(history)
    }
}

fn list_names(root: &Path) -> Result<Vec<String>, String> {
    let mut names = Vec::new();
    for file in fs::read_dir(root).map_err(io_error)? {
        let file = file.map_err(io_error)?;
        if let Ok(name) = file.file_name().into_string() {
            if child(root, &name).is_ok() {
                names.push(name);
            }
        }
    }
    names.sort_unstable();
    Ok(names)
}

pub(super) fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name.len() > 128
        || !name.as_bytes()[0].is_ascii_alphanumeric()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
    {
        return Err("Studio names must use letters, digits, dots, underscores or hyphens".into());
    }
    Ok(())
}

fn child(root: &Path, name: &str) -> Result<PathBuf, String> {
    validate_name(name)?;
    let path = root.join(name).canonicalize().map_err(io_error)?;
    if !path.starts_with(root) || path == root {
        return Err("Studio input is outside its managed directory".into());
    }
    Ok(path)
}

fn io_error(error: impl std::fmt::Display) -> String {
    error.to_string()
}

async fn log_tail(mut input: impl AsyncRead + Unpin) -> String {
    let mut bytes = Vec::new();
    let mut chunk = [0; 4096];
    while let Ok(size @ 1..) = input.read(&mut chunk).await {
        let excess = (bytes.len() + size).saturating_sub(MAX_LOG);
        bytes.drain(..excess);
        bytes.extend_from_slice(&chunk[..size]);
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

enum Outcome {
    Succeeded(Value),
    Failed(String),
    Cancelled,
}
