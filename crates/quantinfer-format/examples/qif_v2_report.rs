//! Native reference report for the fixed unsigned/uncompressed F32 golden profile.
use quantinfer_format::{qif::Reader, sha256_hex};
use std::{env, fs::File, io::Read};

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}
fn u64_at(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: qif_v2_report INPUT.qif")?;
    let mut bytes = Vec::new();
    File::open(path)?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > 1024 * 1024 {
        return Err("golden input exceeds limit".into());
    }
    let reader = Reader::open(&bytes)?;
    let tensor = reader.tensor_info("weight")?;
    if reader.manifest().model_id != "qif-v2-fixture"
        || reader.tensors().len() != 1
        || tensor.codec != 0
        || tensor.dtype != 1
        || tensor.shape != [2]
        || u64_at(&bytes, 128) != 0
    {
        return Err("unsupported golden profile".into());
    }
    reader.verify()?;
    let payload = reader.tensor_data("weight")?;
    if payload != [0, 0, 128, 63, 0, 0, 0, 64] {
        return Err("unsupported golden payload".into());
    }
    let section = |offset| {
        let start = u64_at(&bytes, offset) as usize;
        &bytes[start..start + u64_at(&bytes, offset + 8) as usize]
    };
    let index = section(56);
    if u32_at(index, 12) != 1 {
        return Err("golden profile requires one chunk".into());
    }
    let lengths = [16, 24, 32, 40].map(|offset| u64_at(index, offset));
    let start = 48 + lengths[0] as usize + lengths[1] as usize;
    let record = &index[start..start + 120];
    let chunk = &index[start + 120..start + 192];
    println!(
        "{}",
        serde_json::json!({
            "header_hex":hex(&bytes[..144]),
            "manifest_sha256":sha256_hex(section(24)),
            "metadata_sha256":sha256_hex(section(40)),
            "index_sha256":sha256_hex(index),
            "tensor_sha256":hex(&tensor.checksum_sha256),
            "chunk_sha256":hex(&chunk[40..72]),
            "index_counts":[u32_at(index,8),u32_at(index,12)],
            "table_lengths":lengths,"tensor_id":tensor.tensor_id,
            "name":tensor.name,"shape":tensor.shape,
            "refs":[u32_at(record,8),u32_at(record,12)],
            "encoding":[tensor.dtype,tensor.physical_encoding,tensor.quantization_scheme,tensor.layout,tensor.codec],
            "flags":[u32_at(record,36),u32_at(record,40),u32_at(record,44),u32_at(record,112),u32_at(record,116)],
            "lengths":[tensor.raw_length,tensor.stored_length,u64_at(record,64)],
            "alias_target":u64_at(record,104),
            "chunk":[u32_at(chunk,0) as u64,u32_at(chunk,4) as u64,u64_at(chunk,8),u64_at(chunk,16),u64_at(chunk,24),u64_at(chunk,32)],
            "payload_hex":hex(&payload)
        })
    );
    Ok(())
}
