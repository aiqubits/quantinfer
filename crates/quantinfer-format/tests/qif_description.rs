use quantinfer_format::{qif, GgufReader, GgufValue, GgufWriter};

#[test]
fn gguf_description_matches_published_controls_without_decoding_payload() {
    let mut writer = GgufWriter::new();
    writer.insert_metadata("general.architecture", GgufValue::String("fixture".into()));
    writer
        .add_f32_tensor("weight", vec![2], &[1234.5, -0.375])
        .unwrap();
    let original = writer.finish().unwrap();
    let reader = GgufReader::open(original.clone()).unwrap();
    let (manifest, metadata) = qif::describe_gguf(&reader, "description").unwrap();
    let mut published = Vec::new();
    qif::import_gguf(&reader, "description", &mut published).unwrap();
    let qif = qif::Reader::open(&published).unwrap();
    qif.verify().unwrap();
    assert_eq!(qif.metadata(), &metadata);
    let mut actual = qif.manifest().clone();
    for tensor in &mut actual.tensors {
        tensor.checksum_sha256 = None;
    }
    assert_eq!(
        serde_json::to_value(actual).unwrap(),
        serde_json::to_value(&manifest).unwrap()
    );
    assert!(
        qif::describe_gguf(&GgufReader::open_artifact(published).unwrap(), "duplicate").is_err()
    );
    let payload = reader.tensor_bytes("weight").unwrap();
    let positions: Vec<_> = original
        .windows(payload.len())
        .enumerate()
        .filter_map(|(i, w)| (w == payload).then_some(i))
        .collect();
    assert_eq!(positions.len(), 1);
    let mut damaged = original;
    damaged[positions[0]..positions[0] + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    let damaged = GgufReader::open(damaged).unwrap();
    qif::describe_gguf(&damaged, "description").unwrap();
    assert!(qif::import_gguf(&damaged, "description", &mut Vec::new()).is_err());
}
