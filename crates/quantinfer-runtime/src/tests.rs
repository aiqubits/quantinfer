use super::{argmax, select_token, GenerationConfig, KvCache};
use quantinfer_core::Tensor;

#[test]
fn appends_and_reads_kv_cache() {
    let mut cache = KvCache::new(2, 8, 2, 4).unwrap();
    let key = Tensor::from_f32(vec![1, 2, 4], vec![1.0; 8]).unwrap();
    let value = Tensor::from_f32(vec![1, 2, 4], vec![2.0; 8]).unwrap();
    cache.append(0, &key, &value).unwrap();
    assert_eq!(cache.sequence_length(0).unwrap(), 1);
    assert_eq!(cache.capacity_bytes(), 2 * 8 * 2 * 4 * 2 * 4);
    assert!(cache.memory_bytes() < cache.capacity_bytes());
    assert_eq!(cache.key_values(0).unwrap(), &[1.0; 8]);
    assert_eq!(cache.value_values(0).unwrap(), &[2.0; 8]);
    assert_eq!(cache.keys(0).unwrap().as_f32().unwrap(), &[1.0; 8]);
    assert_eq!(cache.values(0).unwrap().as_f32().unwrap(), &[2.0; 8]);
}

#[test]
fn selects_deterministic_highest_logit() {
    let logits = Tensor::from_f32(vec![4], vec![0.1, 0.9, 0.2, 0.8]).unwrap();
    assert_eq!(argmax(&logits).unwrap(), 1);
    assert_eq!(
        select_token(&logits, &GenerationConfig::default()).unwrap(),
        1
    );
}

#[test]
fn rejects_non_finite_logits() {
    let logits = Tensor::from_f32(vec![2], vec![f32::NAN, 1.0]).unwrap();
    assert!(argmax(&logits).is_err());
    assert!(select_token(&logits, &GenerationConfig::default()).is_err());
}
