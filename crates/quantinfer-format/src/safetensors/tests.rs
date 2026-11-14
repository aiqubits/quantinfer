use super::{SafeTensorDType, SafeTensorFile, SafeTensorShards};

#[test]
fn reads_f32_tensor_and_metadata() {
    let header = br#"{"__metadata__":{"format":"pt"},"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&1.0_f32.to_le_bytes());
    bytes.extend_from_slice(&2.0_f32.to_le_bytes());
    let file = SafeTensorFile::open(bytes).unwrap();
    assert_eq!(file.metadata().get("format"), Some(&"pt".to_owned()));
    assert_eq!(file.tensor_f32("weight").unwrap(), vec![1.0, 2.0]);
}

#[test]
fn rejects_non_finite_floating_tensors() {
    let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&f32::NAN.to_le_bytes());
    let file = SafeTensorFile::open(bytes).unwrap();
    assert!(file.tensor_f32("weight").is_err());
}

#[test]
fn rejects_duplicate_json_tensor_keys() {
    let header = br#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&1.0_f32.to_le_bytes());
    assert!(SafeTensorFile::open(bytes).is_err());
}

#[test]
fn reads_signed_and_unsigned_packed_integer_tensors() {
    let header = br#"{"signed":{"dtype":"I32","shape":[1],"data_offsets":[0,4]},"unsigned":{"dtype":"U32","shape":[1],"data_offsets":[4,8]},"signed64":{"dtype":"I64","shape":[1],"data_offsets":[8,16]},"unsigned64":{"dtype":"U64","shape":[1],"data_offsets":[16,24]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());
    bytes.extend_from_slice(&0x1234_5678_u32.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    bytes.extend_from_slice(&0x1234_5678_9abc_def0_u64.to_le_bytes());
    let file = SafeTensorFile::open(bytes).unwrap();
    assert_eq!(file.tensors()["signed"].dtype, SafeTensorDType::I32);
    assert_eq!(file.tensor_u32("signed").unwrap(), vec![u32::MAX]);
    assert_eq!(file.tensor_u32("unsigned").unwrap(), vec![0x1234_5678]);
    assert_eq!(file.tensor_u64("signed64").unwrap(), vec![u64::MAX]);
    assert_eq!(
        file.tensor_u64("unsigned64").unwrap(),
        vec![0x1234_5678_9abc_def0]
    );
    assert!(file.tensor_f32("signed").is_err());
}

#[test]
fn rejects_tensor_bytes_that_do_not_match_shape() {
    let header = br#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header);
    bytes.extend_from_slice(&1.0_f32.to_le_bytes());
    assert!(SafeTensorFile::open(bytes).is_err());
}

#[test]
fn rejects_oversized_headers_before_allocating_them() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&((64 * 1024 * 1024 + 1) as u64).to_le_bytes());
    assert!(SafeTensorFile::open(bytes).is_err());
}

#[test]
fn loads_tensors_from_a_sharded_index_without_merging_files() {
    let directory = std::env::temp_dir().join(format!("quantinfer-shards-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let write_shard = |name: &str, tensor: &str, value: f32| {
        let header =
            format!(r#"{{"{tensor}":{{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}}}"#);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
        std::fs::write(directory.join(name), bytes).unwrap();
    };
    write_shard("model-00001-of-00002.safetensors", "a", 1.0);
    write_shard("model-00002-of-00002.safetensors", "b", 2.0);
    let index_path = directory.join("model.safetensors.index.json");
    std::fs::write(
            &index_path,
            r#"{"metadata":{"total_size":8},"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
    let shards = SafeTensorShards::open(&index_path).unwrap();
    assert_eq!(
        shards.tensor_shard("a").unwrap(),
        "model-00001-of-00002.safetensors"
    );
    assert_eq!(shards.tensor_f32("b").unwrap(), vec![2.0]);
    assert_eq!(shards.tensors().count(), 2);
    let second = directory.join("model-00002-of-00002.safetensors");
    let checksum = crate::sha256_hex(&std::fs::read(&second).unwrap());
    shards
        .verify_shard_checksum("model-00002-of-00002.safetensors", &checksum)
        .unwrap();
    std::fs::File::options()
        .write(true)
        .open(&second)
        .unwrap()
        .set_len(std::fs::metadata(&second).unwrap().len() - 1)
        .unwrap();
    let error = shards.tensor_f32("b").unwrap_err().to_string();
    assert!(error.contains("model-00002-of-00002.safetensors"));
    assert_eq!(shards.tensor_f32("a").unwrap(), vec![1.0]);
    std::fs::remove_dir_all(directory).unwrap();
}

#[test]
fn rejects_shard_paths_outside_the_index_directory() {
    let directory =
        std::env::temp_dir().join(format!("quantinfer-shard-path-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let index_path = directory.join("model.safetensors.index.json");
    std::fs::write(
        &index_path,
        r#"{"weight_map":{"weight":"../outside.safetensors"}}"#,
    )
    .unwrap();
    let error = SafeTensorShards::open(&index_path).unwrap_err().to_string();
    assert!(error.contains("invalid SafeTensors shard path"));
    std::fs::remove_dir_all(directory).unwrap();
}
