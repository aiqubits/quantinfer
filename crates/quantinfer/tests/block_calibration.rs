#![cfg(feature = "cpu")]
#[path = "support/qif_layout_fixture.rs"]
mod fixture;

use quantinfer::{
    core::Tensor,
    cpu::{resolve_backend, CpuModel},
    format::{qif, GgufReader},
    kernel::KernelBackend,
    model::TensorRole,
    runtime::{DecoderModel, KvCache, Result},
};

fn two_layers(architecture: &str, ggml: bool) -> CpuModel {
    let source = fixture::model(architecture);
    let p = fixture::publication(&source, architecture, ggml, false);
    let mut bytes = Vec::new();
    qif::write_qif(
        p.manifest,
        &p.metadata,
        |n| Ok(p.data[n].as_slice()),
        &mut bytes,
    )
    .unwrap();
    let reader = GgufReader::open_artifact(bytes).unwrap();
    let mut model = CpuModel::from_gguf(&reader, None, None).unwrap();
    let second = CpuModel::from_gguf(&reader, None, None).unwrap();
    match (&mut model, second) {
        (CpuModel::Qwen3(m), CpuModel::Qwen3(mut other)) => {
            m.layers.push(other.layers.remove(0));
            m.config.num_hidden_layers = 2;
        }
        (CpuModel::Llama(m), CpuModel::Llama(mut other)) => {
            m.layers.push(other.layers.remove(0));
            m.config.num_hidden_layers = 2;
        }
        _ => unreachable!(),
    }
    model
}

fn add(a: &Tensor, b: &Tensor) -> Tensor {
    Tensor::from_f32(
        a.shape().clone(),
        a.as_f32()
            .unwrap()
            .iter()
            .zip(b.as_f32().unwrap())
            .map(|(a, b)| a + b)
            .collect(),
    )
    .unwrap()
}

#[test]
fn actual_components_reproduce_full_forward_observations_and_incremental_kv() {
    for architecture in ["qwen3", "llama"] {
        let backend = resolve_backend(&fixture::model(architecture), None).unwrap();
        for ggml in [false, true] {
            let model = two_layers(architecture, ggml);
            let (embedding, norms, final_norm, head, epsilon) = match &model {
                CpuModel::Qwen3(m) => (
                    &m.embedding,
                    m.layers
                        .iter()
                        .map(|l| (&l.input_layernorm, &l.post_attention_layernorm))
                        .collect::<Vec<_>>(),
                    &m.final_norm,
                    &m.lm_head,
                    m.config.rms_norm_eps,
                ),
                CpuModel::Llama(m) => (
                    &m.embedding,
                    m.layers
                        .iter()
                        .map(|l| (&l.input_layernorm, &l.post_attention_layernorm))
                        .collect::<Vec<_>>(),
                    &m.final_norm,
                    &m.lm_head,
                    m.config.rms_norm_eps,
                ),
            };
            let mut normal_cache = model.new_cache(16).unwrap();
            let mut blocks_cache = model.new_cache(16).unwrap();
            let mut trace_cache = model.new_cache(16).unwrap();
            for tokens in [&[1, 2, 3][..], &[4][..]] {
                let reference = model.forward(tokens, &mut normal_cache).unwrap();
                let mut normal_trace = Vec::new();
                model
                    .observe_projections(
                        tokens,
                        &mut trace_cache,
                        &mut |layer: usize, roles: &[TensorRole], input: &Tensor| -> Result<()> {
                            normal_trace.push((layer, roles.to_vec(), input.as_f32()?.to_vec()));
                            Ok(())
                        },
                    )
                    .unwrap();
                let mut blocks_trace = Vec::new();
                let mut observer =
                    |layer: usize, roles: &[TensorRole], input: &Tensor| -> Result<()> {
                        blocks_trace.push((layer, roles.to_vec(), input.as_f32()?.to_vec()));
                        Ok(())
                    };
                let mut hidden = backend.embedding(embedding, tokens).unwrap();
                for (layer, (input_norm, ffn_norm)) in norms.iter().enumerate() {
                    let normalized = backend.rms_norm(&hidden, input_norm, epsilon).unwrap();
                    let attention = model
                        .forward_attention_block(
                            layer,
                            &normalized,
                            &mut blocks_cache,
                            Some(&mut observer),
                        )
                        .unwrap();
                    hidden = add(&hidden, &attention);
                    let normalized = backend.rms_norm(&hidden, ffn_norm, epsilon).unwrap();
                    let ffn = model
                        .forward_feed_forward_block(layer, &normalized, Some(&mut observer))
                        .unwrap();
                    hidden = add(&hidden, &ffn);
                }
                let normalized = backend.rms_norm(&hidden, final_norm, epsilon).unwrap();
                let actual = backend.matmul(&normalized, head).unwrap();
                assert_eq!(
                    actual.as_f32().unwrap(),
                    reference.as_f32().unwrap(),
                    "{architecture}/{ggml}"
                );
                assert_eq!(blocks_trace, normal_trace);
                for layer in 0..2 {
                    assert_eq!(
                        blocks_cache.sequence_length(layer).unwrap(),
                        normal_cache.sequence_length(layer).unwrap()
                    );
                    assert_eq!(
                        blocks_cache.key_values(layer).unwrap(),
                        normal_cache.key_values(layer).unwrap()
                    );
                    assert_eq!(
                        blocks_cache.value_values(layer).unwrap(),
                        normal_cache.value_values(layer).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn block_preflight_rejects_invalid_input_geometry_layer_and_capacity() {
    for architecture in ["qwen3", "llama"] {
        let model = two_layers(architecture, false);
        let input = Tensor::from_f32(vec![2, 8], vec![1.; 16]).unwrap();
        for mut cache in [
            KvCache::new(1, 16, 1, 4).unwrap(),
            KvCache::new(2, 16, 2, 4).unwrap(),
            KvCache::new(2, 16, 1, 2).unwrap(),
            KvCache::new(2, 1, 1, 4).unwrap(),
        ] {
            assert!(model
                .forward_attention_block(0, &input, &mut cache, None)
                .is_err());
            assert_eq!(cache.sequence_length(0).unwrap(), 0);
        }
        for input in [
            Tensor::from_f32(vec![8], vec![1.; 8]).unwrap(),
            Tensor::from_f32(vec![1, 4], vec![1.; 4]).unwrap(),
            Tensor::from_f32(vec![1, 8], vec![f32::NAN; 8]).unwrap(),
        ] {
            let mut cache = model.new_cache(16).unwrap();
            assert!(model
                .forward_attention_block(0, &input, &mut cache, None)
                .is_err());
            assert!(model.forward_feed_forward_block(0, &input, None).is_err());
            assert_eq!(cache.sequence_length(0).unwrap(), 0);
        }
        let mut cache = model.new_cache(16).unwrap();
        assert!(model
            .forward_attention_block(2, &input, &mut cache, None)
            .is_err());
        assert!(model.forward_feed_forward_block(2, &input, None).is_err());
        let mut cache = KvCache::new(2, 64, 1, 4).unwrap();
        let keys = Tensor::from_f32(vec![31, 1, 4], vec![0.; 124]).unwrap();
        cache.append(0, &keys, &keys).unwrap();
        assert!(model
            .forward_attention_block(0, &input, &mut cache, None)
            .is_err());
        assert_eq!(cache.sequence_length(0).unwrap(), 31);
    }
}
