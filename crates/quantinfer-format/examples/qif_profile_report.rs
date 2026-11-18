//! Native oracle for the independent core-profile conformance readers.
use quantinfer_format::{
    qif::{FileReader, MetadataValue, PackageReader, TensorInfo},
    sha256_hex,
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, env, fs, path::Path};

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn metadata(value: &MetadataValue) -> Value {
    match value {
        MetadataValue::F32(value) => json!({"type":"F32","value":hex(&value.to_le_bytes())}),
        MetadataValue::F64(value) => json!({"type":"F64","value":hex(&value.to_le_bytes())}),
        MetadataValue::Array(values) => {
            json!({"type":"Array","value":values.iter().map(metadata).collect::<Vec<_>>()})
        }
        MetadataValue::Map(values) => json!({"type":"Map","value":metadata_map(values)}),
        _ => serde_json::to_value(value).unwrap(),
    }
}
fn metadata_map(values: &BTreeMap<String, MetadataValue>) -> BTreeMap<&str, Value> {
    values
        .iter()
        .map(|(key, value)| (key.as_str(), metadata(value)))
        .collect()
}
fn tensor_report(info: &TensorInfo, record: &[u8], payload: Vec<u8>) -> Value {
    json!({
        "tensor_id":info.tensor_id,"name":info.name,"shape":info.shape,
        "dtype":info.dtype,"physical_encoding":info.physical_encoding,
        "quantization_scheme":info.quantization_scheme,"layout":info.layout,"codec":info.codec,
        "raw_length":info.raw_length,"stored_length":info.stored_length,
        "checksum_sha256":hex(&info.checksum_sha256),"alias_of":info.alias_of,
        "data_offset":u64_at(record,64),"chunk_start":u32_at(record,40),
        "chunk_count":u32_at(record,44),"payload_hex":hex(&payload),
    })
}

const REPORT_LIMIT: u64 = 64 * 1024 * 1024;
fn check_tensor_budget(infos: &[TensorInfo]) -> Result<(), Box<dyn std::error::Error>> {
    let total = infos
        .iter()
        .try_fold(0u64, |sum, info| sum.checked_add(info.raw_length));
    if total.is_none_or(|total| total > REPORT_LIMIT) {
        return Err("reference report tensor budget exceeded".into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    let path = Path::new(
        args.first()
            .ok_or("usage: qif_profile_report INPUT [--trusted-key HEX]...")?,
    );
    let mut trusted = Vec::new();
    let (options, remainder) = args[1..].as_chunks::<2>();
    for pair in options {
        if pair[0] != "--trusted-key"
            || pair[1].len() != 64
            || !pair[1].bytes().all(|v| v.is_ascii_hexdigit())
        {
            return Err("invalid trusted key option".into());
        }
        trusted.push(pair[1].to_ascii_lowercase());
    }
    if !remainder.is_empty() {
        return Err("invalid options".into());
    }
    let (model, values, tensors, index, root, signature) = if path.is_dir() {
        if !trusted.is_empty() {
            return Err("PACKAGE signatures are not supported".into());
        }
        let mut reader = PackageReader::open(path)?;
        let document: Value = serde_json::from_slice(&fs::read(path.join("qif.json"))?)?;
        let mut artifact_bytes = fs::metadata(path.join("qif.json"))?.len();
        for field in ["metadata_path", "index_path"] {
            artifact_bytes = artifact_bytes
                .checked_add(
                    fs::metadata(path.join(document[field].as_str().ok_or("control path")?))?.len(),
                )
                .ok_or("artifact size overflow")?;
        }
        for shard in document["shards"].as_array().ok_or("shard list")? {
            artifact_bytes = artifact_bytes
                .checked_add(shard["size"].as_u64().ok_or("shard size")?)
                .ok_or("artifact size overflow")?;
        }
        if artifact_bytes > REPORT_LIMIT {
            return Err("reference artifact budget exceeded".into());
        }
        check_tensor_budget(reader.tensors())?;
        reader.verify()?;
        let index = fs::read(path.join(document["index_path"].as_str().ok_or("index path")?))?;
        let infos = reader.tensors().to_vec();
        let data = infos
            .iter()
            .map(|info| reader.read_range(&info.name, 0..info.raw_length as usize))
            .collect::<Result<Vec<_>, _>>()?;
        (
            serde_json::to_value(reader.manifest())?,
            reader.metadata().clone(),
            infos.into_iter().zip(data).collect::<Vec<_>>(),
            index,
            document["root_hash"].clone(),
            Value::Null,
        )
    } else {
        if fs::metadata(path)?.len() > REPORT_LIMIT {
            return Err("reference artifact budget exceeded".into());
        }
        let mut reader = FileReader::open(path)?;
        check_tensor_budget(reader.tensors())?;
        reader.verify()?;
        let bytes = fs::read(path)?;
        let start = u64_at(&bytes, 56) as usize;
        let index = bytes[start..start + u64_at(&bytes, 64) as usize].to_vec();
        let signature = if u64_at(&bytes, 128) == 0 {
            if !trusted.is_empty() {
                return Err("a trusted signature is required".into());
            }
            Value::Null
        } else {
            let start = u64_at(&bytes, 120) as usize;
            let key = hex(&bytes[start + 12..start + 44]);
            if !trusted.is_empty() && !trusted.contains(&key) {
                return Err("untrusted signing key".into());
            }
            json!({"public_key":key,"verified":true,"publisher_trust":if trusted.is_empty(){"not_evaluated"}else{"trusted"}})
        };
        let infos = reader.tensors().to_vec();
        let data = infos
            .iter()
            .map(|info| reader.read_range(&info.name, 0..info.raw_length as usize))
            .collect::<Result<Vec<_>, _>>()?;
        (
            serde_json::to_value(reader.manifest())?,
            reader.metadata().clone(),
            infos.into_iter().zip(data).collect::<Vec<_>>(),
            index,
            json!(hex(&bytes[88..120])),
            signature,
        )
    };
    let records_start = 48 + u64_at(&index, 16) as usize + u64_at(&index, 24) as usize;
    let chunks_start = records_start + u64_at(&index, 32) as usize;
    let tensors = tensors
        .into_iter()
        .enumerate()
        .map(|(i, (info, data))| {
            tensor_report(
                &info,
                &index[records_start + i * 120..records_start + (i + 1) * 120],
                data,
            )
        })
        .collect::<Vec<_>>();
    let chunks = index[chunks_start..].as_chunks::<72>().0.iter().map(|chunk| json!({
        "shard_id":u32_at(chunk,0),"codec":u32_at(chunk,4),"raw_offset":u64_at(chunk,8),
        "raw_length":u64_at(chunk,16),"stored_offset":u64_at(chunk,24),"stored_length":u64_at(chunk,32),
        "checksum_sha256":hex(&chunk[40..72]),
    })).collect::<Vec<_>>();
    let metadata_bytes = if path.is_dir() {
        let document: Value = serde_json::from_slice(&fs::read(path.join("qif.json"))?)?;
        fs::read(path.join(document["metadata_path"].as_str().ok_or("metadata path")?))?
    } else {
        let bytes = fs::read(path)?;
        let start = u64_at(&bytes, 40) as usize;
        bytes[start..start + u64_at(&bytes, 48) as usize].to_vec()
    };
    println!(
        "{}",
        json!({"profile":if path.is_dir(){"PACKAGE"}else{"SINGLE"},"root_hash":root,
        "model":model,"metadata":metadata_map(&values),"metadata_sha256":sha256_hex(&metadata_bytes),
        "index_sha256":sha256_hex(&index),"tensors":tensors,"chunks":chunks,"signature":signature})
    );
    Ok(())
}
