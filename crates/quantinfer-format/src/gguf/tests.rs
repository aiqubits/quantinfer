use super::{
    FormatError, GgmlType, GgufReader, GgufSpool, GgufValue, GgufWriter, QUANT_BLOCK_LAYOUT,
};
use quantinfer_core::{
    Q4_0Block, Q6KBlock, Q8_0Block, QuantizedTensor, QuantizedTensor4, QuantizedTensorQ4K,
    QuantizedTensorQ6K, Tensor,
};

#[test]
fn round_trips_scalar_metadata_and_f32_tensor() {
    let mut writer = GgufWriter::new();
    writer.insert_metadata(
        "general.architecture",
        GgufValue::String("qwen3".to_owned()),
    );
    writer.insert_metadata("qwen3.context_length", GgufValue::U32(128));
    writer
        .add_f32_tensor("token_embd.weight", vec![2, 2], &[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    writer
        .add_f16_tensor("f16.weight", vec![2], &[1.0, 2.0])
        .unwrap();
    writer
        .add_bf16_tensor("bf16.weight", vec![2], &[3.0, 4.0])
        .unwrap();
    let bytes = writer.finish().unwrap();
    let reader = GgufReader::open(bytes).unwrap();
    assert_eq!(
        reader.metadata().get("general.architecture"),
        Some(&GgufValue::String("qwen3".to_owned()))
    );
    assert_eq!(reader.tensors()[0].shape, vec![2, 2]);
    assert_eq!(reader.tensor_bytes("token_embd.weight").unwrap().len(), 16);
    assert_eq!(
        reader.tensor_f32("token_embd.weight").unwrap(),
        Tensor::from_f32(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap()
    );
    assert_eq!(
        reader.tensor_f32("f16.weight").unwrap(),
        Tensor::from_f32(vec![2], vec![1.0, 2.0]).unwrap()
    );
    assert_eq!(
        reader.tensor_f32("bf16.weight").unwrap(),
        Tensor::from_f32(vec![2], vec![3.0, 4.0]).unwrap()
    );
}

#[test]
fn rejects_zero_dimension_gguf_tensors_before_quantized_decode() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&4_u64.to_le_bytes());
    bytes.extend_from_slice(b"zero");
    bytes.extend_from_slice(&1_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&(GgmlType::Q8_0 as u32).to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.resize(bytes.len().next_multiple_of(32), 0);
    assert!(matches!(
        GgufReader::open(bytes),
        Err(FormatError::Invalid(message)) if message.contains("dimensions")
    ));
}

#[test]
fn round_trips_metadata_arrays_and_q8_0_tensor() {
    let mut writer = GgufWriter::new();
    writer.insert_metadata(
        "tokenizer.ggml.tokens",
        GgufValue::Array(vec![
            GgufValue::String("<|endoftext|>".to_owned()),
            GgufValue::String("hello".to_owned()),
        ]),
    );
    let quantized = QuantizedTensor::from_blocks(
        vec![32],
        vec![Q8_0Block {
            scale: 0.25,
            values: [1_i8; 32],
        }],
    )
    .unwrap();
    writer.add_q8_0_tensor("blk.0.attn_q.weight", &quantized);
    let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
    assert_eq!(
        reader.metadata().get("tokenizer.ggml.tokens"),
        Some(&GgufValue::Array(vec![
            GgufValue::String("<|endoftext|>".to_owned()),
            GgufValue::String("hello".to_owned()),
        ]))
    );
    assert_eq!(
        reader.tensor_info("blk.0.attn_q.weight").unwrap().dtype,
        GgmlType::Q8_0
    );
    assert_eq!(
        reader.tensor_q8_0("blk.0.attn_q.weight").unwrap(),
        quantized
    );
}

#[test]
fn round_trips_q4_0_tensor_with_ggml_packing() {
    let mut values = [0_u8; 32];
    for (index, value) in values.iter_mut().enumerate() {
        *value = index.min(31 - index) as u8;
    }
    let quantized =
        QuantizedTensor4::from_blocks(vec![32], vec![Q4_0Block { scale: 0.5, values }]).unwrap();
    let mut writer = GgufWriter::new();
    writer.add_q4_0_tensor("blk.0.attn_q.weight", &quantized);
    let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
    assert_eq!(
        reader.tensor_info("blk.0.attn_q.weight").unwrap().dtype,
        GgmlType::Q4_0
    );
    assert_eq!(
        reader.tensor_bytes("blk.0.attn_q.weight").unwrap(),
        [
            0x00, 0x38, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b,
            0x3c, 0x2d, 0x1e, 0x0f
        ],
    );
    assert_eq!(
        reader.tensor_q4_0("blk.0.attn_q.weight").unwrap(),
        quantized
    );
}

#[test]
fn spools_tensors_with_identical_metadata_and_alignment() {
    let quantized = QuantizedTensor4::from_blocks(
        vec![32],
        vec![Q4_0Block {
            scale: 0.5,
            values: [8; 32],
        }],
    )
    .unwrap();
    let mut writer = GgufWriter::new();
    writer.insert_metadata("general.architecture", GgufValue::String("llama".into()));
    let mut spool = GgufSpool::new(std::io::Cursor::new(Vec::new()), writer.clone());
    let add_f32 = |writer: &mut GgufWriter| writer.add_f32_tensor("norm", vec![1], &[2.0]);
    let add_q4 = |writer: &mut GgufWriter| {
        writer.add_q4_0_tensor("projection", &quantized);
        Ok(())
    };
    add_f32(&mut writer).unwrap();
    add_q4(&mut writer).unwrap();
    spool.append(add_f32).unwrap();
    spool.append(add_q4).unwrap();
    let mut output = Vec::new();
    spool.finish(&mut output).unwrap();
    assert_eq!(output, writer.finish().unwrap());
    let reader = GgufReader::open(output).unwrap();
    assert_eq!(reader.tensor_info("projection").unwrap().offset, 32);
    assert_eq!(reader.tensor_q4_0("projection").unwrap(), quantized);
}

#[test]
fn reads_standard_and_legacy_quant_block_bytes() {
    let q4_values = std::array::from_fn(|index| (index % 16) as u8);
    let q4 = QuantizedTensor4::from_blocks(
        vec![32],
        vec![Q4_0Block {
            scale: 0.5,
            values: q4_values,
        }],
    )
    .unwrap();
    let q6 = QuantizedTensorQ6K::from_blocks(
        vec![256],
        vec![Q6KBlock {
            scale: 0.5,
            scales: [-2; 16],
            low_bits: [0x12; 128],
            high_bits: [0x34; 64],
        }],
    )
    .unwrap();
    let mut writer = GgufWriter::new();
    writer.insert_metadata(
        "general.architecture",
        GgufValue::String("llama".to_owned()),
    );
    writer.add_q4_0_tensor("q4", &q4);
    writer.add_q6_k_tensor("q6", &q6);
    let standard_q6 = [&[0x12; 128][..], &[0x34; 64], &[0xfe; 16], &[0x00, 0x38]].concat();
    writer.metadata.remove(QUANT_BLOCK_LAYOUT);
    let reader = GgufReader::open(writer.clone().finish().unwrap()).unwrap();
    assert_eq!(reader.tensor_bytes("q6").unwrap(), standard_q6);
    assert_eq!(reader.tensor_q4_0("q4").unwrap(), q4);
    assert_eq!(reader.tensor_q6_k("q6").unwrap(), q6);

    writer.insert_metadata("llama.tie_word_embeddings", GgufValue::Bool(true));
    writer.tensors[0].data = vec![
        0x00, 0x38, 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x10, 0x32, 0x54, 0x76, 0x98,
        0xba, 0xdc, 0xfe,
    ];
    writer.tensors[1].data = [
        &standard_q6[208..],
        &standard_q6[192..208],
        &standard_q6[..192],
    ]
    .concat();
    let legacy = GgufReader::open(writer.finish().unwrap()).unwrap();
    assert_eq!(legacy.tensor_q4_0("q4").unwrap(), q4);
    assert_eq!(legacy.tensor_q6_k("q6").unwrap(), q6);
}

#[test]
fn round_trips_q4_k_tensor_with_standard_ggml_layout() {
    let quantized = QuantizedTensorQ4K::from_blocks(
        vec![256],
        vec![quantinfer_core::Q4KBlock {
            scale: 0.125,
            min_scale: 0.25,
            scales: [63; 12],
            values: [0x21; 128],
        }],
    )
    .unwrap();
    let mut writer = GgufWriter::new();
    writer.add_q4_k_tensor("blk.0.attn_q.weight", &quantized);
    let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
    let restored: QuantizedTensorQ4K = reader.tensor_q4_k("blk.0.attn_q.weight").unwrap();
    assert_eq!(restored, quantized);
    assert_eq!(reader.tensor_info("blk.0.attn_q.weight").unwrap().size, 144);
}

#[test]
fn rejects_zero_alignment_in_untrusted_gguf() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
    bytes.extend_from_slice(&1_u64.to_le_bytes());
    bytes.extend_from_slice(&17_u64.to_le_bytes());
    bytes.extend_from_slice(b"general.alignment");
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u32.to_le_bytes());
    assert!(matches!(
        GgufReader::open(bytes),
        Err(FormatError::Invalid(_))
    ));
}

#[test]
fn reads_all_standard_scalar_metadata_kinds() {
    // The reader-level coverage for the compact scalar kinds is exercised
    // through a hand-built metadata section because GgufWriter emits the
    // canonical wide kinds only.
    let mut raw = Vec::new();
    raw.extend_from_slice(b"GGUF");
    raw.extend_from_slice(&3_u32.to_le_bytes());
    raw.extend_from_slice(&0_u64.to_le_bytes());
    raw.extend_from_slice(&5_u64.to_le_bytes());
    for (name, kind, value) in [
        ("u8", 0_u32, vec![7]),
        ("i8", 1_u32, vec![0xfb]),
        ("u16", 2_u32, 0x1234_u16.to_le_bytes().to_vec()),
        ("i16", 3_u32, (-7_i16).to_le_bytes().to_vec()),
        ("f64", 12_u32, 1.25_f64.to_le_bytes().to_vec()),
    ] {
        raw.extend_from_slice(&(name.len() as u64).to_le_bytes());
        raw.extend_from_slice(name.as_bytes());
        raw.extend_from_slice(&kind.to_le_bytes());
        raw.extend_from_slice(&value);
    }
    let reader = GgufReader::open(raw).unwrap();
    assert_eq!(reader.metadata().get("f64"), Some(&GgufValue::F64(1.25)));
}

#[test]
fn rejects_duplicate_tensor_names() {
    let mut raw = Vec::new();
    raw.extend_from_slice(b"GGUF");
    raw.extend_from_slice(&3_u32.to_le_bytes());
    raw.extend_from_slice(&2_u64.to_le_bytes());
    raw.extend_from_slice(&0_u64.to_le_bytes());
    for _ in 0..2 {
        raw.extend_from_slice(&3_u64.to_le_bytes());
        raw.extend_from_slice(b"dup");
        raw.extend_from_slice(&0_u32.to_le_bytes());
        raw.extend_from_slice(&0_u32.to_le_bytes());
        raw.extend_from_slice(&0_u64.to_le_bytes());
    }
    assert!(matches!(
        GgufReader::open(raw),
        Err(FormatError::Invalid(message)) if message.contains("duplicate GGUF tensor name")
    ));
}

#[test]
fn rejects_duplicate_metadata_keys_and_non_finite_tensors() {
    let mut raw = Vec::new();
    raw.extend_from_slice(b"GGUF");
    raw.extend_from_slice(&3_u32.to_le_bytes());
    raw.extend_from_slice(&0_u64.to_le_bytes());
    raw.extend_from_slice(&2_u64.to_le_bytes());
    for value in [1_u32, 2_u32] {
        raw.extend_from_slice(&3_u64.to_le_bytes());
        raw.extend_from_slice(b"dup");
        raw.extend_from_slice(&4_u32.to_le_bytes());
        raw.extend_from_slice(&value.to_le_bytes());
    }
    assert!(matches!(
        GgufReader::open(raw),
        Err(FormatError::Invalid(message)) if message.contains("metadata keys must be unique")
    ));

    let mut writer = GgufWriter::new();
    writer
        .add_f32_tensor("non_finite", vec![1], &[f32::NAN])
        .unwrap();
    let reader = GgufReader::open(writer.finish().unwrap()).unwrap();
    assert!(matches!(
        reader.tensor_f32("non_finite"),
        Err(FormatError::Invalid(message)) if message.contains("non-finite")
    ));
}
