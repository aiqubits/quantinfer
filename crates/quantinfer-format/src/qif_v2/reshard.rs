use std::collections::BTreeMap;

use quantinfer_core::TensorStorageDescriptor;

use crate::{GgmlType, ModelManifest, Result};

use super::invalid;

/// Deterministic raw-byte shard placement for a QIF2-PACKAGE.
///
/// The plan describes logical placement only. Writers remain free to account
/// for codec expansion and alignment while preserving the tensor order and
/// descriptors recorded here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QifV2ReshardPlan {
    pub max_shard_bytes: usize,
    pub tensor_shards: BTreeMap<String, Vec<u32>>,
    pub shard_raw_bytes: Vec<u64>,
}

/// Plan a deterministic sequential resharding of a validated model manifest.
/// A tensor larger than the limit is represented on multiple consecutive
/// shards; no tensor descriptor or logical encoding is changed.
pub fn plan_qif_v2_reshard(
    manifest: &ModelManifest,
    max_shard_bytes: usize,
) -> Result<QifV2ReshardPlan> {
    if max_shard_bytes == 0 {
        return Err(invalid("reshard shard size must be greater than zero"));
    }
    manifest.validate()?;
    let mut shard_raw_bytes = vec![0_u64];
    let mut tensor_shards = BTreeMap::new();
    for tensor in &manifest.tensors {
        let mut remaining = payload_size(&tensor.descriptor)? as u64;
        let mut assigned = Vec::new();
        while remaining > 0 {
            let shard_id = (shard_raw_bytes.len() - 1) as u32;
            let current = shard_raw_bytes
                .last_mut()
                .ok_or_else(|| invalid("reshard plan has no current shard"))?;
            if *current == max_shard_bytes as u64 {
                shard_raw_bytes.push(0);
                continue;
            }
            let available = max_shard_bytes as u64 - *current;
            let take = remaining.min(available);
            if assigned.last().copied() != Some(shard_id) {
                assigned.push(shard_id);
            }
            *current = current
                .checked_add(take)
                .ok_or_else(|| invalid("reshard shard size overflows"))?;
            remaining -= take;
        }
        tensor_shards.insert(tensor.name.clone(), assigned);
    }
    Ok(QifV2ReshardPlan {
        max_shard_bytes,
        tensor_shards,
        shard_raw_bytes,
    })
}

fn payload_size(descriptor: &TensorStorageDescriptor) -> Result<usize> {
    descriptor.validate()?;
    let dtype = match descriptor.dtype {
        quantinfer_core::DType::F32 => GgmlType::F32,
        quantinfer_core::DType::F16 => GgmlType::F16,
        quantinfer_core::DType::BF16 => GgmlType::BF16,
        quantinfer_core::DType::Q8_0 => GgmlType::Q8_0,
        quantinfer_core::DType::Q4_0 => GgmlType::Q4_0,
        quantinfer_core::DType::Q4_K => GgmlType::Q4_K,
        quantinfer_core::DType::Q5_K => GgmlType::Q5_K,
        quantinfer_core::DType::Q6_K => GgmlType::Q6_K,
    };
    crate::tensor_byte_size(dtype, descriptor.logical_shape.dims())
}
