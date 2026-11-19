//! On-demand QIF storage shared by SINGLE and PACKAGE runtime readers.
use std::{
    borrow::Cow,
    collections::BTreeMap,
    fmt,
    io::{Read, Seek, SeekFrom},
    ops::Range,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
};

use super::{
    chunk::Chunk,
    invalid, metadata_to_gguf,
    payload::PayloadVerifier,
    reader::{dtype_from_id, dtype_to_ggml},
    FileReader, MetadataValue, PackageReader, TensorInfo, CODEC_NONE,
};
use crate::{gguf::PackedTensorReader, FormatError, GgufTensorInfo, ModelManifest, Result};

enum Source {
    Single(FileReader),
    Package(PackageReader),
}

struct Control<'a> {
    manifest: &'a ModelManifest,
    metadata: &'a BTreeMap<String, MetadataValue>,
    tensors: &'a [TensorInfo],
    chunks: &'a [Chunk],
    artifact_length: u64,
    control_length: usize,
}

impl Source {
    fn control(&self) -> Control<'_> {
        match self {
            Self::Single(reader) => Control {
                manifest: reader.manifest(),
                metadata: reader.metadata(),
                tensors: reader.tensors(),
                chunks: &reader.chunks,
                artifact_length: reader.artifact_length,
                control_length: reader.control_length,
            },
            Self::Package(reader) => Control {
                manifest: reader.manifest(),
                metadata: reader.metadata(),
                tensors: reader.tensors(),
                chunks: &reader.chunks,
                artifact_length: reader.artifact_length,
                control_length: reader.control_length,
            },
        }
    }

    fn read_range(&mut self, name: &str, range: Range<usize>) -> Result<Vec<u8>> {
        match self {
            Self::Single(reader) => reader.read_range(name, range),
            Self::Package(reader) => reader.read_range(name, range),
        }
    }

    fn verify(&mut self) -> Result<()> {
        match self {
            Self::Single(reader) => reader.verify(),
            Self::Package(reader) => reader.verify(),
        }
    }
}

struct Slot {
    name: String,
    dtype: u32,
    length: usize,
    hash: [u8; 32],
    cached: OnceLock<Result<Vec<u8>>>,
}

pub(crate) struct FileBacking {
    source: Mutex<Source>,
    slots: Vec<Slot>,
    names: BTreeMap<String, usize>,
    pub(crate) artifact_length: usize,
    pub(crate) control_length: usize,
    pub(crate) loading_workspace: usize,
}

impl fmt::Debug for FileBacking {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QifFileBacking")
            .field("physical_tensors", &self.slots.len())
            .field("cached_bytes", &self.cached_bytes())
            .finish()
    }
}

impl FileBacking {
    fn slot(&self, name: &str) -> Result<&Slot> {
        self.names
            .get(name)
            .map(|&slot| &self.slots[slot])
            .ok_or_else(|| invalid(format!("unknown tensor {name}")))
    }

    fn load(&self, slot: &Slot) -> Result<Vec<u8>> {
        let bytes = self
            .source
            .lock()
            .map_err(|_| invalid("file reader lock poisoned"))?
            .read_range(&slot.name, 0..slot.length)?;
        let mut verifier = PayloadVerifier::new(dtype_from_id(slot.dtype)?, &slot.name);
        verifier.update(&bytes)?;
        verifier.finish(slot.length as u64, &slot.hash)?;
        Ok(bytes)
    }

    pub(crate) fn data(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        let slot = self.slot(name)?;
        match slot.cached.get() {
            Some(Ok(bytes)) => Ok(Cow::Borrowed(bytes)),
            Some(Err(error)) => Err(error.clone()),
            None => self.load(slot).map(Cow::Owned),
        }
    }

    pub(crate) fn borrowed(&self, name: &str) -> Result<&[u8]> {
        let slot = self.slot(name)?;
        match slot.cached.get_or_init(|| self.load(slot)) {
            Ok(bytes) => Ok(bytes),
            Err(error) => Err(error.clone()),
        }
    }

    pub(crate) fn cached_bytes(&self) -> usize {
        self.slots
            .iter()
            .filter_map(|slot| slot.cached.get()?.as_ref().ok())
            .map(Vec::len)
            .sum()
    }

    pub(crate) fn verify(&self) -> Result<()> {
        self.source
            .lock()
            .map_err(|_| invalid("file reader lock poisoned"))?
            .verify()
    }
}

pub(crate) fn open_packed_path(path: &Path, budget: Option<usize>) -> Result<PackedTensorReader> {
    let source = if path.is_dir() {
        Source::Package(PackageReader::open(path)?)
    } else {
        let resolved = std::fs::canonicalize(path).map_err(io_error)?;
        let file = crate::manifest::open_readonly_nofollow(&resolved).map_err(io_error)?;
        return open_packed_file(file, budget);
    };
    project(source, budget)
}

pub(crate) fn open_packed_file(
    mut file: std::fs::File,
    budget: Option<usize>,
) -> Result<PackedTensorReader> {
    file.seek(SeekFrom::Start(0)).map_err(io_error)?;
    let mut magic = [0; 8];
    file.read_exact(&mut magic).map_err(io_error)?;
    if !super::has_magic(&magic) {
        if &magic[..4] != b"GGUF" {
            return Err(invalid("unsupported artifact magic/version"));
        }
        let length = file.metadata().map_err(io_error)?.len();
        if budget.is_some_and(|limit| length > limit as u64) {
            return Err(invalid("model input exceeds memory budget"));
        }
        file.seek(SeekFrom::Start(0)).map_err(io_error)?;
        let mut bytes = Vec::new();
        if let Some(limit) = budget {
            file.take((limit as u64).saturating_add(1))
                .read_to_end(&mut bytes)
                .map_err(io_error)?;
        } else {
            file.read_to_end(&mut bytes).map_err(io_error)?;
        }
        return PackedTensorReader::open_artifact_with_memory_budget(bytes, budget);
    }
    project(Source::Single(FileReader::from_file(file)?), budget)
}

fn project(source: Source, budget: Option<usize>) -> Result<PackedTensorReader> {
    let controls = source.control();
    let ggml = match controls.metadata.get(super::GGML_DIMENSION_ORDER) {
        Some(MetadataValue::Bool(value)) => *value,
        Some(_) => return Err(invalid("GGML dimension order metadata must be boolean")),
        None => false,
    };
    let mut slots = Vec::new();
    let mut names = BTreeMap::new();
    let mut targets = Vec::<usize>::new();
    let mut offsets = Vec::<u64>::new();
    let mut raw_length = 0usize;
    let mut tensors = Vec::new();
    let mut largest = 0usize;
    for tensor in controls.tensors {
        let length = usize::try_from(tensor.raw_length)
            .map_err(|_| invalid("tensor length overflows usize"))?;
        let (slot, offset) = if let Some(target) = tensor.alias_of {
            let target =
                usize::try_from(target).map_err(|_| invalid("alias id overflows usize"))?;
            (targets[target], offsets[target])
        } else {
            let slot = slots.len();
            let offset = raw_length as u64;
            raw_length = raw_length
                .checked_add(length)
                .ok_or_else(|| invalid("weights size overflows usize"))?;
            largest = largest.max(length);
            slots.push(Slot {
                name: tensor.name.clone(),
                dtype: tensor.dtype,
                length,
                hash: tensor.checksum_sha256,
                cached: OnceLock::new(),
            });
            (slot, offset)
        };
        targets.push(slot);
        offsets.push(offset);
        names.insert(tensor.name.clone(), slot);
        let mut shape = tensor
            .shape
            .iter()
            .map(|&value| {
                usize::try_from(value).map_err(|_| invalid("tensor dimension overflows usize"))
            })
            .collect::<Result<Vec<_>>>()?;
        if ggml {
            shape.reverse();
        }
        tensors.push(GgufTensorInfo {
            name: tensor.name.clone(),
            shape,
            dtype: dtype_to_ggml(tensor.dtype)?,
            offset,
            size: length,
        });
    }
    let required = raw_length
        .checked_add(controls.control_length)
        .ok_or_else(|| invalid("weights and controls size overflow"))?;
    if let Some(limit) = budget.filter(|&limit| required > limit) {
        return Err(invalid(format!(
            "weights and controls need at least {required} bytes, memory budget is {limit}"
        )));
    }
    let chunk_workspace = controls
        .chunks
        .iter()
        .filter(|chunk| chunk.codec != CODEC_NONE)
        .try_fold(0usize, |max, chunk| {
            let length = chunk
                .raw_length
                .checked_mul(2)?
                .checked_add(chunk.stored_length)?;
            Some(max.max(usize::try_from(length).ok()?))
        })
        .ok_or_else(|| invalid("chunk workspace size overflows"))?;
    let backing = FileBacking {
        slots,
        names,
        artifact_length: usize::try_from(controls.artifact_length)
            .map_err(|_| invalid("artifact size overflows usize"))?,
        control_length: controls.control_length,
        loading_workspace: largest
            .checked_add(chunk_workspace)
            .ok_or_else(|| invalid("loading workspace overflows"))?,
        source: Mutex::new(source),
    };
    // Borrow controls only until all owned projections have been created.
    let source = backing
        .source
        .lock()
        .map_err(|_| invalid("file reader lock poisoned"))?;
    let control = source.control();
    let metadata = metadata_to_gguf(control.metadata)?;
    let manifest = control.manifest.clone();
    let native = control.metadata.clone();
    drop(source);
    Ok(PackedTensorReader {
        bytes: Vec::new(),
        data_start: 0,
        metadata,
        tensors,
        embedded_manifest: Some(manifest),
        qif_metadata: Some(native),
        file_backing: Some(Arc::new(backing)),
    })
}

fn io_error(error: std::io::Error) -> FormatError {
    FormatError::Io(format!("cannot read model artifact: {error}"))
}
