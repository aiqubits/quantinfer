use quantinfer_format::{GgufReader, GgufValue, GgufWriter};

/// Two non-symmetric layers; head_dim=8 makes the Llama Q/K permutation
/// non-self-inverse, unlike the four-channel fixture.
pub fn model(architecture: &str, ggml: bool) -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.insert_metadata(
        "general.architecture",
        GgufValue::String(architecture.into()),
    );
    for (suffix, value) in [
        ("vocab_size", 16),
        ("context_length", 32),
        ("embedding_length", 64),
        ("feed_forward_length", 128),
        ("block_count", 2),
        ("attention.head_count", 8),
        ("attention.head_count_kv", 2),
        ("attention.head_dim", 8),
        ("attention.key_length", 8),
        ("attention.value_length", 8),
        ("rope.dimension_count", 8),
    ] {
        writer.insert_metadata(format!("{architecture}.{suffix}"), GgufValue::U32(value));
    }
    if !ggml {
        writer.insert_metadata(
            format!("{architecture}.tie_word_embeddings"),
            GgufValue::Bool(true),
        );
    }
    writer.insert_metadata("tokenizer.ggml.pre", GgufValue::String("gpt2".into()));
    writer.insert_metadata("tokenizer.ggml.eos_token_id", GgufValue::U32(0));
    writer.insert_metadata(
        "tokenizer.ggml.tokens",
        GgufValue::Array(
            [
                "<eos>", "a", "b", "Ġ", "x", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
            ]
            .map(|s| GgufValue::String(s.into()))
            .to_vec(),
        ),
    );
    let mut tensors = vec![
        ("token_embd.weight".into(), vec![16, 64]),
        ("output_norm.weight".into(), vec![64]),
    ];
    for layer in 0..2 {
        for (suffix, shape) in [
            ("attn_norm", vec![64]),
            ("attn_q", vec![64, 64]),
            ("attn_k", vec![64, 16]),
            ("attn_v", vec![64, 16]),
            ("attn_output", vec![64, 64]),
            ("ffn_norm", vec![64]),
            ("ffn_gate", vec![64, 128]),
            ("ffn_up", vec![64, 128]),
            ("ffn_down", vec![128, 64]),
        ] {
            tensors.push((format!("blk.{layer}.{suffix}.weight"), shape));
        }
        if architecture == "qwen3" {
            for suffix in ["attn_q_norm", "attn_k_norm"] {
                tensors.push((format!("blk.{layer}.{suffix}.weight"), vec![8]));
            }
        }
    }
    for (seed, (name, mut shape)) in tensors.into_iter().enumerate() {
        let values: Vec<f32> = (0..shape.iter().product::<usize>())
            .map(|i| {
                if name.contains("norm") {
                    1.
                } else {
                    ((i * 17 + i / 7 + seed * 19 + 5) % 97) as f32 * 0.00073 - 0.035
                }
            })
            .collect();
        let physical = if ggml && shape.len() == 2 && name != "token_embd.weight" {
            let (k, n) = (shape[0], shape[1]);
            let mut result = vec![0.; values.len()];
            for output in 0..n {
                let channel = if architecture == "llama"
                    && (name.ends_with("attn_q.weight") || name.ends_with("attn_k.weight"))
                {
                    quantinfer_format::qk_channel(output, 8)
                } else {
                    output
                };
                for input in 0..k {
                    result[output * k + input] = values[input * n + channel];
                }
            }
            result
        } else {
            values
        };
        // GGUF projection dimensions are [input,output] even though payload
        // rows are output/input; the embedding uses [hidden,vocabulary].
        if ggml && name == "token_embd.weight" {
            shape.reverse();
        }
        writer.add_f32_tensor(name, shape, &physical).unwrap();
    }
    let bytes = writer.finish().unwrap();
    assert_eq!(
        GgufReader::open(bytes.clone())
            .unwrap()
            .uses_ggml_output_rows()
            .unwrap(),
        ggml
    );
    bytes
}
