use quantinfer::format::{
    qif::{self, MetadataValue},
    GgufReader, GgufValue, GgufWriter, ModelManifest,
};
use std::collections::BTreeMap;

pub fn model(architecture: &str) -> GgufReader {
    let mut writer = GgufWriter::new();
    writer.insert_metadata(
        "general.architecture",
        GgufValue::String(architecture.into()),
    );
    for (suffix, value) in [
        ("vocab_size", 16),
        ("context_length", 32),
        ("embedding_length", 8),
        ("feed_forward_length", 16),
        ("block_count", 1),
        ("attention.head_count", 2),
        ("attention.head_count_kv", 1),
        ("attention.head_dim", 4),
    ] {
        writer.insert_metadata(format!("{architecture}.{suffix}"), GgufValue::U32(value));
    }
    writer.insert_metadata(
        format!("{architecture}.tie_word_embeddings"),
        GgufValue::Bool(false),
    );
    let mut shapes = vec![
        ("token_embd.weight", vec![16, 8]),
        ("output.weight", vec![8, 16]),
        ("output_norm.weight", vec![8]),
        ("blk.0.attn_norm.weight", vec![8]),
        ("blk.0.attn_q.weight", vec![8, 8]),
        ("blk.0.attn_k.weight", vec![8, 4]),
        ("blk.0.attn_v.weight", vec![8, 4]),
        ("blk.0.attn_output.weight", vec![8, 8]),
        ("blk.0.ffn_norm.weight", vec![8]),
        ("blk.0.ffn_gate.weight", vec![8, 16]),
        ("blk.0.ffn_up.weight", vec![8, 16]),
        ("blk.0.ffn_down.weight", vec![16, 8]),
    ];
    if architecture == "qwen3" {
        shapes.extend([
            ("blk.0.attn_q_norm.weight", vec![4]),
            ("blk.0.attn_k_norm.weight", vec![4]),
        ]);
    }
    for (seed, (name, shape)) in shapes.into_iter().enumerate() {
        let values: Vec<_> = (0..shape.iter().product())
            .map(|i| {
                if name.contains("norm") {
                    1.0
                } else {
                    ((i * 17 + i / 3 + seed * 19 + 5) % 29) as f32 * 0.03 - 0.4
                }
            })
            .collect();
        writer.add_f32_tensor(name, shape, &values).unwrap();
    }
    GgufReader::open(writer.finish().unwrap()).unwrap()
}

pub struct Publication {
    pub manifest: ModelManifest,
    pub metadata: BTreeMap<String, MetadataValue>,
    pub data: BTreeMap<String, Vec<u8>>,
}

pub fn publication(
    source: &GgufReader,
    architecture: &str,
    ggml: bool,
    mixed: bool,
) -> Publication {
    let mut encoded = Vec::new();
    qif::import_gguf(source, "layout-model", &mut encoded).unwrap();
    let original = qif::Reader::open(&encoded).unwrap();
    let mut manifest = original.manifest().clone();
    let mut metadata = original.metadata().clone();
    metadata.insert(
        "quantinfer.qif.ggml_dimension_order".into(),
        MetadataValue::Bool(ggml),
    );
    let mut data = BTreeMap::new();
    for tensor in &mut manifest.tensors {
        let shape = tensor.descriptor.logical_shape.dims().to_vec();
        let name = tensor.name.as_str();
        let output_rows = mixed
            && (name == "token_embd.weight"
                || name == "output.weight"
                || name.ends_with("attn_output.weight")
                || name.ends_with("attn_k.weight")
                || name.ends_with("ffn_up.weight"));
        tensor.descriptor.physical_layout = if output_rows {
            "ggml_output_rows"
        } else {
            "row_major"
        }
        .into();
        let logical = source.tensor_f32(name).unwrap();
        let values = logical.as_f32().unwrap();
        let stored = if shape.len() == 2 {
            let embedding = name == "token_embd.weight";
            if ggml && !embedding {
                tensor.descriptor.logical_shape = vec![shape[1], shape[0]].into();
            }
            // Native embedding/table bytes use their logical axes. GGML
            // projection logical axes are [N,K], with physical output rows.
            let transpose = if ggml { !embedding } else { output_rows };
            if transpose {
                let mut physical = vec![0.0; values.len()];
                for o in 0..shape[1] {
                    let native_o = if ggml
                        && architecture == "llama"
                        && (name.ends_with("attn_q.weight") || name.ends_with("attn_k.weight"))
                    {
                        o / 4 * 4 + [0, 2, 1, 3][o % 4]
                    } else {
                        o
                    };
                    for i in 0..shape[0] {
                        physical[o * shape[0] + i] = values[i * shape[1] + native_o];
                    }
                }
                physical
            } else {
                values.to_vec()
            }
        } else {
            values.to_vec()
        };
        tensor.checksum_sha256 = None;
        data.insert(
            name.to_owned(),
            stored.iter().flat_map(|v| v.to_le_bytes()).collect(),
        );
    }
    Publication {
        manifest,
        metadata,
        data,
    }
}
