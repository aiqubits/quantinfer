//! QuantInfer conversion command-line entry point.

mod common;
mod convert;
mod model;
mod policy;
mod source;
mod tokenizer;

use convert::run;

fn main() {
    if let Err(error) = run() {
        eprintln!("quantinfer-convert: {error}");
        std::process::exit(1);
    }
}

pub(crate) fn print_help() {
    println!(
        "Usage: quantinfer-convert --config config.json --weights model.safetensors|model.safetensors.index.json --output model.gguf [--weights-checksums checksums.sha256] [--architecture qwen3|llama] [--tokenizer tokenizer.json] [--manifest manifest.json] [--quantize f32|q8_0|q4_0|q4_k|q5_k|q6_k] [--mixed-precision PLAN.json|--mixed-precision-policy POLICY.json] [--imatrix CHANNEL_WEIGHTS.json|IMATRIX.gguf] [--ggml] [--gguf-layout quantinfer|ggml] [--tokenizer-pre NAME]\n       quantinfer-convert from-gguf --input model.gguf --output model.qif [--model-id NAME]"
    );
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::common::load_imatrix;
    use super::model::ConversionModel;
    use super::tokenizer::load_tokenizer_metadata;
    use quantinfer_format::{GgufValue, GgufWriter};

    #[test]
    fn reads_dense_llama_imatrix_and_normalizes_observations() {
        let mut writer = GgufWriter::new();
        writer.insert_metadata("general.type", GgufValue::String("imatrix".to_owned()));
        writer
            .add_f32_tensor("blk.0.attn_q.weight.in_sum2", vec![2, 1], &[4.0, 12.0])
            .unwrap();
        writer
            .add_f32_tensor("blk.0.attn_q.weight.counts", vec![1], &[4.0])
            .unwrap();
        let weights = load_imatrix(writer.finish().unwrap()).unwrap();
        assert_eq!(weights["blk.0.attn_q.weight"], [1.0, 3.0]);
        assert!(load_imatrix(GgufWriter::new().finish().unwrap()).is_err());
    }

    #[test]
    #[cfg(feature = "ggml")]
    fn selects_the_external_q8_provider_explicitly() {
        use super::policy::q8_provider;
        use quantinfer_core::Tensor;
        use quantinfer_quant::QuantizedData;

        let zeros = Tensor::from_f32(vec![32], vec![0.0; 32]).unwrap();
        for (use_ggml, expected_scale) in [(false, 1.0), (true, 0.0)] {
            let data = q8_provider(use_ggml).unwrap().quantize(&zeros).unwrap();
            let QuantizedData::Q8_0(quantized) = data else {
                panic!("expected Q8_0 blocks");
            };
            assert_eq!(quantized.blocks()[0].scale, expected_scale);
        }
    }

    #[test]
    fn reads_huggingface_tokenizer_vocab_and_merges() {
        let directory = std::env::temp_dir();
        let path = directory.join(format!("quantinfer-tokenizer-{}.json", std::process::id()));
        std::fs::write(
            &path,
            r#"{
                "model": {
                    "vocab": {"h": 0, "i": 1, "hi": 2},
                    "merges": [["h", "i"]]
                },
                "added_tokens": [{"id": 3, "content": "<|im_end|>", "special": true}]
            }"#,
        )
        .unwrap();
        let metadata = load_tokenizer_metadata(
            &path,
            r#"{"vocab_size": 4, "bos_token_id": 0, "eos_token_id": 3}"#,
        )
        .unwrap();
        std::fs::remove_file(path).unwrap();
        assert_eq!(metadata.tokens, vec!["h", "i", "hi", "<|im_end|>"]);
        assert_eq!(metadata.merges, vec!["h i"]);
        assert_eq!(metadata.eos_token_id, Some(3));
    }

    #[test]
    fn detects_llama_config_and_validates_canonical_tensor_set() {
        let model = ConversionModel::from_config(
            r#"{
                "model_type": "llama",
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64,
                "tie_word_embeddings": true,
                "architectures": ["LlamaForCausalLM"]
            }"#,
            None,
        )
        .unwrap();
        assert_eq!(model.architecture(), "llama");
        assert_eq!(model.vocab_size(), 32);
        assert_eq!(
            model.canonical_tensor_name("model.layers.0.self_attn.q_proj.weight"),
            Some("blk.0.attn_q.weight".to_owned())
        );
        let names = [
            "token_embd.weight",
            "output_norm.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
        model.validate_required_tensors(&names).unwrap();
    }

    #[test]
    fn rejects_config_without_an_explicit_supported_architecture() {
        let error = ConversionModel::from_config(
            r#"{
                "vocab_size": 32,
                "hidden_size": 16,
                "intermediate_size": 32,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "max_position_embeddings": 64
            }"#,
            None,
        )
        .unwrap_err();
        assert!(error.contains("model architecture is missing"));
    }
}
