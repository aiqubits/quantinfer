use quantinfer_runtime::{DecodeBatchItem, DecoderModel, KvCache};

pub fn assert_matches_independent_sequences(
    model: &impl DecoderModel,
    prefetched: &KvCache,
    empty: &KvCache,
    next_token: usize,
    fresh_prompt: &[usize],
    layers: usize,
    vocab: usize,
) {
    let decode_token = [next_token];
    let mut independent_decode = prefetched.clone();
    let mut independent_prefill = empty.clone();
    let serial_decode = model
        .forward(&decode_token, &mut independent_decode)
        .unwrap();
    let serial_prefill = model
        .forward(fresh_prompt, &mut independent_prefill)
        .unwrap();
    let mut batched_decode = prefetched.clone();
    let mut batched_prefill = empty.clone();
    let mut invalid = empty.clone();
    let outputs = model.forward_batch(&mut [
        DecodeBatchItem {
            tokens: &decode_token,
            cache: &mut batched_decode,
        },
        DecodeBatchItem {
            tokens: &[usize::MAX],
            cache: &mut invalid,
        },
        DecodeBatchItem {
            tokens: fresh_prompt,
            cache: &mut batched_prefill,
        },
    ]);
    assert_eq!(outputs.len(), 3);
    assert!(outputs[1].is_err());
    for (index, serial, batched_cache, independent_cache) in [
        (0, &serial_decode, &batched_decode, &independent_decode),
        (2, &serial_prefill, &batched_prefill, &independent_prefill),
    ] {
        let serial = serial.as_f32().unwrap();
        let batch_row = outputs[index].as_ref().unwrap().as_f32().unwrap();
        assert_eq!(batch_row, &serial[serial.len() - vocab..]);
        for layer in 0..layers {
            assert_eq!(
                batched_cache.key_values(layer).unwrap(),
                independent_cache.key_values(layer).unwrap()
            );
            assert_eq!(
                batched_cache.value_values(layer).unwrap(),
                independent_cache.value_values(layer).unwrap()
            );
            assert_eq!(invalid.sequence_length(layer).unwrap(), 0);
        }
    }
    println!("mixed prefill and decode batch matches independent logits and KV values");
}
