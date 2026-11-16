use std::collections::BTreeMap;
use std::fs::{self, File};
use std::hint::black_box;
use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::qif::{
    write_qif, FileReader as QifFileReader, MetadataValue, Reader as QifReader,
};
use quantinfer_format::{GgufReader, ModelManifest, SafeTensorFileReader, TensorManifest};

#[derive(Clone, Copy, Default)]
struct ResourceSnapshot {
    rss_kb: Option<u64>,
    minor_faults: Option<u64>,
    major_faults: Option<u64>,
}

pub(crate) fn run() {
    let qif = super::option_value("--qif");
    let safetensors = super::option_value("--safetensors");
    let gguf = super::option_value("--gguf");
    let tensor = super::option_value("--tensor");
    if qif.is_some() || safetensors.is_some() || gguf.is_some() {
        run_external(
            qif.as_deref(),
            safetensors.as_deref(),
            gguf.as_deref(),
            tensor.as_deref(),
        );
    } else {
        run_synthetic();
    }
}

fn run_synthetic() {
    let payload = (0..(1024 * 1024 / 4))
        .flat_map(|value| (value as f32).to_le_bytes())
        .collect::<Vec<_>>();
    let mut manifest = ModelManifest::new("qif-benchmark", "fixture");
    manifest.tensors.push(TensorManifest {
        name: "weight".to_owned(),
        descriptor: TensorStorageDescriptor::floating(
            DType::F32,
            Shape::from(vec![payload.len() / 4]),
        )
        .unwrap(),
        checksum_sha256: None,
    });
    let metadata = BTreeMap::<String, MetadataValue>::new();
    let mut artifact = Vec::new();
    write_qif(
        manifest,
        &metadata,
        |_| Ok(payload.as_slice()),
        &mut artifact,
    )
    .expect("benchmark artifact should be valid");
    let started = Instant::now();
    let reader = QifReader::open(&artifact).expect("benchmark reader should open");
    println!("qif_open_ns={}", started.elapsed().as_nanos());
    let started = Instant::now();
    black_box(
        reader
            .read_range("weight", 0..4096)
            .expect("range should read"),
    );
    println!("qif_first_range_ns={}", started.elapsed().as_nanos());
    let started = Instant::now();
    reader.verify().expect("benchmark verification should pass");
    println!("qif_full_verify_ns={}", started.elapsed().as_nanos());
    println!("qif_artifact_bytes={}", artifact.len());
}

fn run_external(
    qif_path: Option<&str>,
    safetensors_path: Option<&str>,
    gguf_path: Option<&str>,
    tensor_name: Option<&str>,
) {
    if let Some(path) = qif_path {
        benchmark_qif(path, tensor_name);
    }
    if let Some(path) = safetensors_path {
        benchmark_safetensors(path, tensor_name);
    }
    if let Some(path) = gguf_path {
        benchmark_gguf(path, tensor_name);
    }
}

fn benchmark_qif(path: &str, requested_tensor: Option<&str>) {
    let before = resources();
    let started = Instant::now();
    let mut reader = QifFileReader::open(path).expect("QIF file should open");
    let open_ns = started.elapsed().as_nanos();
    let name = requested_tensor
        .map(str::to_owned)
        .unwrap_or_else(|| reader.tensors()[0].name.clone());
    let info = reader.tensor_info(&name).expect("QIF2 tensor should exist");
    let length = usize::try_from(info.raw_length).expect("QIF2 tensor length should fit usize");
    let started = Instant::now();
    black_box(
        reader
            .read_range(&name, 0..length.min(4096))
            .expect("QIF2 range should read"),
    );
    let range_ns = started.elapsed().as_nanos();
    let started = Instant::now();
    reader.verify().expect("QIF2 verification should pass");
    let verify_ns = started.elapsed().as_nanos();
    let after = resources();
    let bytes = fs::metadata(path).expect("QIF2 file should stat").len();
    println!("qif_tensor={name}");
    println!("qif_open_ns={open_ns}");
    println!("qif_first_range_ns={range_ns}");
    println!("qif_full_verify_ns={verify_ns}");
    println!("qif_artifact_bytes={bytes}");
    print_resource_delta("qif", before, after);
}

fn benchmark_safetensors(path: &str, requested_tensor: Option<&str>) {
    let before = resources();
    let started = Instant::now();
    let reader = SafeTensorFileReader::open(path).expect("SafeTensors file should open");
    let open_ns = started.elapsed().as_nanos();
    let name = requested_tensor.map(str::to_owned).unwrap_or_else(|| {
        reader
            .tensors()
            .keys()
            .next()
            .expect("tensor exists")
            .clone()
    });
    let info = reader
        .tensors()
        .get(&name)
        .expect("SafeTensors tensor should exist");
    let file_len = fs::metadata(path)
        .expect("SafeTensors file should stat")
        .len();
    let mut file = File::open(path).expect("SafeTensors file should open for range");
    let header_size = read_u64(&mut file).expect("SafeTensors header should read") as usize;
    let start = 8usize + header_size + info.data_offsets.0;
    let length = (info.data_offsets.1 - info.data_offsets.0).min(4096);
    file.seek(SeekFrom::Start(start as u64))
        .expect("SafeTensors range should seek");
    let started = Instant::now();
    let mut bytes = vec![0_u8; length];
    file.read_exact(&mut bytes)
        .expect("SafeTensors range should read");
    black_box(bytes);
    let range_ns = started.elapsed().as_nanos();
    let after = resources();
    println!("safetensors_tensor={name}");
    println!("safetensors_open_ns={open_ns}");
    println!("safetensors_first_range_ns={range_ns}");
    println!("safetensors_artifact_bytes={file_len}");
    println!("safetensors_verify_supported=0");
    print_resource_delta("safetensors", before, after);
}

fn benchmark_gguf(path: &str, requested_tensor: Option<&str>) {
    let before = resources();
    let file_bytes = fs::read(path).expect("GGUF file should read");
    let started = Instant::now();
    let reader = GgufReader::open(file_bytes).expect("GGUF file should open");
    let open_ns = started.elapsed().as_nanos();
    let name = requested_tensor
        .map(str::to_owned)
        .unwrap_or_else(|| reader.tensors()[0].name.clone());
    let started = Instant::now();
    let bytes = reader.tensor_bytes(&name).expect("GGUF tensor should read");
    black_box(&bytes[..bytes.len().min(4096)]);
    let range_ns = started.elapsed().as_nanos();
    let after = resources();
    println!("gguf_tensor={name}");
    println!("gguf_open_ns={open_ns}");
    println!("gguf_first_range_ns={range_ns}");
    println!(
        "gguf_artifact_bytes={}",
        fs::metadata(path).expect("GGUF file should stat").len()
    );
    println!("gguf_verify_supported=0");
    print_resource_delta("gguf", before, after);
}

fn read_u64(file: &mut File) -> std::io::Result<u64> {
    let mut bytes = [0_u8; 8];
    file.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn resources() -> ResourceSnapshot {
    #[cfg(target_os = "linux")]
    {
        let rss_kb = fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|text| {
                text.lines()
                    .find(|line| line.starts_with("VmRSS:"))
                    .and_then(|line| line.split_whitespace().nth(1))
                    .and_then(|value| value.parse().ok())
            });
        let faults = fs::read_to_string("/proc/self/stat")
            .ok()
            .and_then(|text| text.rsplit_once(") ").map(|(_, rest)| rest.to_owned()))
            .and_then(|rest| {
                let fields = rest.split_whitespace().collect::<Vec<_>>();
                Some((fields.get(7)?.parse().ok()?, fields.get(9)?.parse().ok()?))
            });
        return ResourceSnapshot {
            rss_kb,
            minor_faults: faults.map(|value| value.0),
            major_faults: faults.map(|value| value.1),
        };
    }
    #[allow(unreachable_code)]
    ResourceSnapshot::default()
}

fn print_resource_delta(prefix: &str, before: ResourceSnapshot, after: ResourceSnapshot) {
    if let (Some(start), Some(end)) = (before.rss_kb, after.rss_kb) {
        println!("{prefix}_rss_kb_delta={}", end.saturating_sub(start));
    }
    if let (Some(start), Some(end)) = (before.minor_faults, after.minor_faults) {
        println!(
            "{prefix}_minor_page_faults_delta={}",
            end.saturating_sub(start)
        );
    }
    if let (Some(start), Some(end)) = (before.major_faults, after.major_faults) {
        println!(
            "{prefix}_major_page_faults_delta={}",
            end.saturating_sub(start)
        );
    }
}
