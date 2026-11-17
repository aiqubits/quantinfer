use std::collections::BTreeMap;
use std::fs;
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::qif::{
    write_qif, FileReader as QifFileReader, MetadataValue, Reader as QifReader,
};
use quantinfer_format::{GgufReader, ModelManifest, SafeTensorFileReader, TensorManifest};

#[derive(Clone, Copy, Default)]
struct ResourceSnapshot {
    rss_kb: Option<u64>,
    peak_rss_kb: Option<u64>,
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
    println!("qif_benchmark_mode=synthetic_memory");
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
    println!("benchmark_process_isolation=1");
    println!("benchmark_cache_state=uncontrolled");
    for (format, path) in [
        ("qif", qif_path),
        ("safetensors", safetensors_path),
        ("gguf", gguf_path),
    ] {
        let Some(path) = path else {
            continue;
        };
        let name = super::option_value(&format!("--{format}-tensor"))
            .or_else(|| tensor_name.map(str::to_owned));
        let mut command =
            Command::new(std::env::current_exe().expect("benchmark executable exists"));
        command.args(["qif-bench-worker", "--format", format, "--input", path]);
        if let Some(name) = name {
            command.args(["--tensor", &name]);
        }
        assert!(
            command
                .status()
                .expect("benchmark worker should start")
                .success(),
            "{format} benchmark worker failed"
        );
    }
}

pub(crate) fn run_worker() {
    let format = super::option_value("--format").expect("worker requires --format");
    let path = super::option_value("--input").expect("worker requires --input");
    let name = super::option_value("--tensor");
    println!("{format}_process_id={}", std::process::id());
    println!(
        "{format}_tensor_selection={}",
        if name.is_some() {
            "explicit"
        } else {
            "format_default"
        }
    );
    match format.as_str() {
        "qif" => benchmark_qif(&path, name.as_deref()),
        "safetensors" => benchmark_safetensors(&path, name.as_deref()),
        "gguf" => benchmark_gguf(&path, name.as_deref()),
        _ => panic!("unsupported benchmark format {format}"),
    }
}

fn print_range_report(
    prefix: &str,
    name: &str,
    dtype: &str,
    bytes: &[u8],
    timings: (u128, u128, u128),
    before: ResourceSnapshot,
    after: ResourceSnapshot,
) {
    println!("{prefix}_tensor={name}");
    println!("{prefix}_dtype={dtype}");
    println!("{prefix}_open_ns={}", timings.0);
    println!("{prefix}_first_range_ns={}", timings.1);
    println!("{prefix}_open_to_first_range_ns={}", timings.2);
    println!("{prefix}_range_bytes={}", bytes.len());
    println!(
        "{prefix}_range_sha256={}",
        quantinfer_format::sha256_hex(bytes)
    );
    print_resource_delta(prefix, before, after);
}

fn benchmark_qif(path: &str, requested_tensor: Option<&str>) {
    println!("qif_open_strategy=file_backed_control");
    let before = resources();
    let total_started = Instant::now();
    let mut reader = QifFileReader::open(path).expect("QIF file should open");
    let open_ns = total_started.elapsed().as_nanos();
    let name = requested_tensor
        .map(str::to_owned)
        .unwrap_or_else(|| reader.tensors()[0].name.clone());
    let info = reader
        .tensor_info(&name)
        .expect("QIF tensor should exist")
        .clone();
    let length = usize::try_from(info.raw_length).expect("QIF tensor size fits usize");
    let started = Instant::now();
    let bytes = reader
        .read_range(&name, 0..length.min(4096))
        .expect("QIF range should read");
    black_box(&bytes);
    let range_ns = started.elapsed().as_nanos();
    let total_ns = total_started.elapsed().as_nanos();
    let after = resources();
    let dtype = format!(
        "{:?}",
        reader
            .manifest()
            .tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
            .descriptor
            .dtype
    );
    print_range_report(
        "qif",
        &name,
        &dtype,
        &bytes,
        (open_ns, range_ns, total_ns),
        before,
        after,
    );
    println!("qif_tensor_raw_bytes={}", info.raw_length);
    println!("qif_physical_encoding_id={}", info.physical_encoding);
    println!("qif_codec_id={}", info.codec);
    println!("qif_layout_id={}", info.layout);
    println!(
        "qif_artifact_bytes={}",
        fs::metadata(path).expect("QIF file should stat").len()
    );
    println!("qif_verify_supported=1");
    let verify_before = resources();
    let started = Instant::now();
    reader.verify().expect("QIF verification should pass");
    let verify_ns = started.elapsed().as_nanos();
    let verify_after = resources();
    println!("qif_full_verify_ns={verify_ns}");
    print_resource_delta("qif_verify", verify_before, verify_after);
}

fn benchmark_safetensors(path: &str, requested_tensor: Option<&str>) {
    println!("safetensors_open_strategy=file_backed_control");
    let before = resources();
    let total_started = Instant::now();
    let reader = SafeTensorFileReader::open(path).expect("SafeTensors file should open");
    let open_ns = total_started.elapsed().as_nanos();
    let name = requested_tensor.map(str::to_owned).unwrap_or_else(|| {
        reader
            .tensors()
            .keys()
            .next()
            .expect("tensor exists")
            .clone()
    });
    let info = &reader.tensors()[&name];
    let length = info.data_offsets.1 - info.data_offsets.0;
    let started = Instant::now();
    let bytes = reader
        .read_range(&name, 0..length.min(4096))
        .expect("SafeTensors range should read");
    black_box(&bytes);
    let range_ns = started.elapsed().as_nanos();
    let total_ns = total_started.elapsed().as_nanos();
    let after = resources();
    print_range_report(
        "safetensors",
        &name,
        &format!("{:?}", info.dtype),
        &bytes,
        (open_ns, range_ns, total_ns),
        before,
        after,
    );
    println!("safetensors_tensor_raw_bytes={length}");
    println!(
        "safetensors_artifact_bytes={}",
        fs::metadata(path)
            .expect("SafeTensors file should stat")
            .len()
    );
    println!("safetensors_verify_supported=0");
}

fn benchmark_gguf(path: &str, requested_tensor: Option<&str>) {
    println!("gguf_open_strategy=eager_file");
    let before = resources();
    let total_started = Instant::now();
    let file_bytes = fs::read(path).expect("GGUF file should read");
    let reader = GgufReader::open(file_bytes).expect("GGUF file should open");
    let open_ns = total_started.elapsed().as_nanos();
    let name = requested_tensor
        .map(str::to_owned)
        .unwrap_or_else(|| reader.tensors()[0].name.clone());
    let info = reader.tensor_info(&name).expect("GGUF tensor should exist");
    let started = Instant::now();
    let payload = reader.tensor_bytes(&name).expect("GGUF tensor should read");
    let bytes = payload[..payload.len().min(4096)].to_vec();
    black_box(&bytes);
    let range_ns = started.elapsed().as_nanos();
    let total_ns = total_started.elapsed().as_nanos();
    let after = resources();
    print_range_report(
        "gguf",
        &name,
        &format!("{:?}", info.dtype),
        &bytes,
        (open_ns, range_ns, total_ns),
        before,
        after,
    );
    println!("gguf_tensor_raw_bytes={}", info.size);
    println!(
        "gguf_artifact_bytes={}",
        fs::metadata(path).expect("GGUF file should stat").len()
    );
    println!("gguf_verify_supported=0");
}

fn resources() -> ResourceSnapshot {
    #[cfg(target_os = "linux")]
    {
        let status = fs::read_to_string("/proc/self/status").ok();
        let memory = |key: &str| {
            status
                .as_ref()?
                .lines()
                .find(|line| line.starts_with(key))?
                .split_whitespace()
                .nth(1)?
                .parse()
                .ok()
        };
        let rss_kb = memory("VmRSS:");
        let peak_rss_kb = memory("VmHWM:");
        let faults = fs::read_to_string("/proc/self/stat")
            .ok()
            .and_then(|text| text.rsplit_once(") ").map(|(_, rest)| rest.to_owned()))
            .and_then(|rest| {
                let fields = rest.split_whitespace().collect::<Vec<_>>();
                Some((fields.get(7)?.parse().ok()?, fields.get(9)?.parse().ok()?))
            });
        return ResourceSnapshot {
            rss_kb,
            peak_rss_kb,
            minor_faults: faults.map(|value| value.0),
            major_faults: faults.map(|value| value.1),
        };
    }
    #[allow(unreachable_code)]
    ResourceSnapshot::default()
}

fn print_resource_delta(prefix: &str, before: ResourceSnapshot, after: ResourceSnapshot) {
    if let Some(peak) = after.peak_rss_kb {
        println!("{prefix}_peak_rss_kb={peak}");
    }
    if let (Some(start), Some(end)) = (before.rss_kb, after.rss_kb) {
        println!(
            "{prefix}_rss_kb_delta={}",
            i128::from(end) - i128::from(start)
        );
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
