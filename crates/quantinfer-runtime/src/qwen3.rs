use quantinfer_core::{DType, QuantizedTensor, Shape, Tensor};
use quantinfer_kernel::KernelBackend;
use quantinfer_model::Qwen3Config;

use crate::{KvCache, Result, RuntimeError};

#[derive(Debug, Clone)]
pub enum LinearWeight {
    F32(Tensor),
    Q8_0(QuantizedTensor),
}

impl LinearWeight {
    pub fn shape(&self) -> &Shape {
        match self {
            Self::F32(tensor) => tensor.shape(),
            Self::Q8_0(tensor) => tensor.shape(),
        }
    }

    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::Q8_0(_) => DType::Q8_0,
        }
    }

    fn matmul<B: KernelBackend>(&self, backend: &B, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::F32(weight) => Ok(backend.matmul(input, weight)?),
            Self::Q8_0(weight) => Ok(backend.matmul_q8_0(input, weight)?),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3LayerWeights {
    pub input_layernorm: Tensor,
    pub q_proj: LinearWeight,
    pub q_norm: Tensor,
    pub k_proj: LinearWeight,
    pub k_norm: Tensor,
    pub v_proj: LinearWeight,
    pub o_proj: LinearWeight,
    pub post_attention_layernorm: Tensor,
    pub gate_proj: LinearWeight,
    pub up_proj: LinearWeight,
    pub down_proj: LinearWeight,
}

impl Qwen3LayerWeights {
    pub fn validate(&self, config: &Qwen3Config) -> Result<()> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let query = config.query_projection_size();
        let key_value = config.key_value_projection_size();
        let expected = [
            (
                "input_layernorm",
                self.input_layernorm.shape().dims(),
                &[hidden][..],
            ),
            ("q_proj", self.q_proj.shape().dims(), &[hidden, query][..]),
            (
                "q_norm",
                self.q_norm.shape().dims(),
                &[config.effective_head_dim()][..],
            ),
            (
                "k_proj",
                self.k_proj.shape().dims(),
                &[hidden, key_value][..],
            ),
            (
                "k_norm",
                self.k_norm.shape().dims(),
                &[config.effective_head_dim()][..],
            ),
            (
                "v_proj",
                self.v_proj.shape().dims(),
                &[hidden, key_value][..],
            ),
            ("o_proj", self.o_proj.shape().dims(), &[query, hidden][..]),
            (
                "post_attention_layernorm",
                self.post_attention_layernorm.shape().dims(),
                &[hidden][..],
            ),
            (
                "gate_proj",
                self.gate_proj.shape().dims(),
                &[hidden, intermediate][..],
            ),
            (
                "up_proj",
                self.up_proj.shape().dims(),
                &[hidden, intermediate][..],
            ),
            (
                "down_proj",
                self.down_proj.shape().dims(),
                &[intermediate, hidden][..],
            ),
        ];
        for (name, actual, expected) in expected {
            if actual != expected {
                return Err(RuntimeError::InvalidInput(format!(
                    "Qwen3 {name} has shape {actual:?}, expected {expected:?}"
                )));
            }
            if !matches!(
                name,
                "q_proj" | "k_proj" | "v_proj" | "o_proj" | "gate_proj" | "up_proj" | "down_proj"
            ) && self.weight_dtype(name)? != DType::F32
            {
                return Err(RuntimeError::InvalidInput(format!(
                    "Qwen3 {name} must use F32 for the reference runtime"
                )));
            }
        }
        Ok(())
    }

    fn weight_dtype(&self, name: &str) -> Result<DType> {
        match name {
            "input_layernorm" => Ok(self.input_layernorm.dtype()),
            "q_proj" => Ok(self.q_proj.dtype()),
            "q_norm" => Ok(self.q_norm.dtype()),
            "k_proj" => Ok(self.k_proj.dtype()),
            "k_norm" => Ok(self.k_norm.dtype()),
            "v_proj" => Ok(self.v_proj.dtype()),
            "o_proj" => Ok(self.o_proj.dtype()),
            "post_attention_layernorm" => Ok(self.post_attention_layernorm.dtype()),
            "gate_proj" => Ok(self.gate_proj.dtype()),
            "up_proj" => Ok(self.up_proj.dtype()),
            "down_proj" => Ok(self.down_proj.dtype()),
            _ => Err(RuntimeError::InvalidInput(format!(
                "unknown Qwen3 weight {name}"
            ))),
        }
    }

    fn forward<B: KernelBackend>(
        &self,
        backend: &B,
        config: &Qwen3Config,
        hidden: &Tensor,
        layer_index: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let sequence_length = hidden.shape().dims()[0];
        let residual = hidden.clone();
        let normalized = backend.rms_norm(hidden, &self.input_layernorm, config.rms_norm_eps)?;
        let query = self
            .q_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_attention_heads,
                config.effective_head_dim(),
            ]))?;
        let key = self
            .k_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_key_value_heads,
                config.effective_head_dim(),
            ]))?;
        let value = self
            .v_proj
            .matmul(backend, &normalized)?
            .reshape(Shape::from(vec![
                sequence_length,
                config.num_key_value_heads,
                config.effective_head_dim(),
            ]))?;
        let query = backend.rms_norm_heads(&query, &self.q_norm, config.rms_norm_eps)?;
        let key = backend.rms_norm_heads(&key, &self.k_norm, config.rms_norm_eps)?;
        let start_position = cache.sequence_length(layer_index)?;
        let (cos, sin) = rope_tables(
            config.rope_theta,
            config.effective_head_dim(),
            start_position,
            sequence_length,
        );
        let query = backend.apply_rope(&query, &cos, &sin)?;
        let key = backend.apply_rope(&key, &cos, &sin)?;
        cache.append(layer_index, &key, &value)?;
        let key_values = cache.key_values(layer_index)?;
        let value_values = cache.value_values(layer_index)?;
        let key_tokens = cache.sequence_length(layer_index)?;
        let attended = gqa_attention(
            &query,
            key_values,
            value_values,
            key_tokens,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.effective_head_dim(),
            start_position,
        )?;
        let attended = attended.reshape(Shape::from(vec![
            sequence_length,
            config.query_projection_size(),
        ]))?;
        let projected = self.o_proj.matmul(backend, &attended)?;
        let residual = add(&residual, &projected)?;
        let normalized = backend.rms_norm(
            &residual,
            &self.post_attention_layernorm,
            config.rms_norm_eps,
        )?;
        let gate = self.gate_proj.matmul(backend, &normalized)?;
        let up = self.up_proj.matmul(backend, &normalized)?;
        let feed_forward = backend.silu_mul(&gate, &up)?;
        let feed_forward = self.down_proj.matmul(backend, &feed_forward)?;
        add(&residual, &feed_forward)
    }
}

pub struct Qwen3Model<B> {
    pub config: Qwen3Config,
    backend: B,
    pub embedding: Tensor,
    pub layers: Vec<Qwen3LayerWeights>,
    pub final_norm: Tensor,
    pub lm_head: Tensor,
}

impl<B: KernelBackend> Qwen3Model<B> {
    pub fn new(
        config: Qwen3Config,
        backend: B,
        embedding: Tensor,
        layers: Vec<Qwen3LayerWeights>,
        final_norm: Tensor,
        lm_head: Tensor,
    ) -> Result<Self> {
        config.validate().map_err(RuntimeError::InvalidInput)?;
        expect_shape(
            &embedding,
            &[config.vocab_size, config.hidden_size],
            "embedding",
        )?;
        expect_shape(&final_norm, &[config.hidden_size], "final_norm")?;
        expect_shape(
            &lm_head,
            &[config.hidden_size, config.vocab_size],
            "lm_head",
        )?;
        if layers.len() != config.num_hidden_layers {
            return Err(RuntimeError::InvalidInput(format!(
                "Qwen3 has {} layers but received {} weight sets",
                config.num_hidden_layers,
                layers.len()
            )));
        }
        for layer in &layers {
            layer.validate(&config)?;
        }
        Ok(Self {
            config,
            backend,
            embedding,
            layers,
            final_norm,
            lm_head,
        })
    }

    pub fn forward(&self, token_ids: &[usize], cache: &mut KvCache) -> Result<Tensor> {
        if token_ids.is_empty() {
            return Err(RuntimeError::InvalidInput(
                "Qwen3 forward requires at least one token".to_owned(),
            ));
        }
        if cache.sequence_length(0)? + token_ids.len() > self.config.max_position_embeddings {
            return Err(RuntimeError::InvalidInput(
                "input exceeds Qwen3 max_position_embeddings".to_owned(),
            ));
        }
        let mut hidden = self.backend.embedding(&self.embedding, token_ids)?;
        for (index, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&self.backend, &self.config, &hidden, index, cache)?;
        }
        let hidden = self
            .backend
            .rms_norm(&hidden, &self.final_norm, self.config.rms_norm_eps)?;
        Ok(self.backend.matmul(&hidden, &self.lm_head)?)
    }
}

fn expect_shape(tensor: &Tensor, expected: &[usize], name: &str) -> Result<()> {
    if tensor.shape().dims() != expected {
        return Err(RuntimeError::InvalidInput(format!(
            "Qwen3 {name} has shape {:?}, expected {expected:?}",
            tensor.shape().dims()
        )));
    }
    if tensor.dtype() != quantinfer_core::DType::F32 {
        return Err(RuntimeError::InvalidInput(format!(
            "Qwen3 {name} must use F32 for the reference runtime"
        )));
    }
    Ok(())
}

fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    if lhs.shape() != rhs.shape() {
        return Err(RuntimeError::InvalidInput(
            "residual tensors must have identical shapes".to_owned(),
        ));
    }
    let values = lhs
        .as_f32()?
        .iter()
        .zip(rhs.as_f32()?)
        .map(|(left, right)| left + right)
        .collect();
    Ok(Tensor::from_f32(lhs.shape().clone(), values)?)
}

fn rope_tables(theta: f32, head_dim: usize, start: usize, length: usize) -> (Vec<f32>, Vec<f32>) {
    let pair_count = head_dim / 2;
    let mut cos = Vec::with_capacity(length * pair_count);
    let mut sin = Vec::with_capacity(length * pair_count);
    for position in start..start + length {
        for pair in 0..pair_count {
            let exponent = (2 * pair) as f32 / head_dim as f32;
            let angle = position as f32 * theta.powf(-exponent);
            cos.push(angle.cos());
            sin.push(angle.sin());
        }
    }
    (cos, sin)
}

fn gqa_attention(
    query: &Tensor,
    key_values: &[f32],
    value_values: &[f32],
    key_tokens: usize,
    query_heads: usize,
    key_value_heads: usize,
    head_dim: usize,
    start_position: usize,
) -> Result<Tensor> {
    let query_dims = query.shape().dims();
    if query_dims.len() != 3
        || query_dims[1] != query_heads
        || query_dims[2] != head_dim
        || key_tokens == 0
        || key_value_heads == 0
        || key_values.len() != key_tokens * key_value_heads * head_dim
        || value_values.len() != key_values.len()
        || !query_heads.is_multiple_of(key_value_heads)
    {
        return Err(RuntimeError::InvalidInput(
            "invalid Qwen3 GQA tensor shapes".to_owned(),
        ));
    }
    let query_values = query.as_f32()?;
    let query_tokens = query_dims[0];
    if key_tokens < start_position + query_tokens {
        return Err(RuntimeError::InvalidInput(
            "KV cache does not contain all causal attention keys".to_owned(),
        ));
    }
    let group_size = query_heads / key_value_heads;
    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0; query_tokens * query_heads * head_dim];
    for token in 0..query_tokens {
        let last_key = start_position + token;
        for query_head in 0..query_heads {
            let kv_head = query_head / group_size;
            let query_start = (token * query_heads + query_head) * head_dim;
            let mut scores = Vec::with_capacity(last_key + 1);
            for key_token in 0..=last_key {
                let key_start = (key_token * key_value_heads + kv_head) * head_dim;
                let score = query_values[query_start..query_start + head_dim]
                    .iter()
                    .zip(&key_values[key_start..key_start + head_dim])
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    * scale;
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .sum::<f32>();
            let output_start = query_start;
            for (key_token, score) in scores.iter().copied().enumerate() {
                let weight = (score - maximum).exp() / denominator;
                let value_start = (key_token * key_value_heads + kv_head) * head_dim;
                for dimension in 0..head_dim {
                    output[output_start + dimension] +=
                        weight * value_values[value_start + dimension];
                }
            }
        }
    }
    Ok(Tensor::from_f32(
        Shape::from(vec![query_tokens, query_heads, head_dim]),
        output,
    )?)
}

#[cfg(test)]
mod tests {
    use super::{LinearWeight, Qwen3LayerWeights, Qwen3Model};
    use crate::KvCache;
    use quantinfer_backend_cpu::CpuScalar;
    use quantinfer_core::Tensor;
    use quantinfer_model::Qwen3Config;
    use quantinfer_quant::quantize_q8_0;

    fn filled(shape: Vec<usize>, value: f32) -> Tensor {
        let count = shape.iter().product();
        Tensor::from_f32(shape, vec![value; count]).unwrap()
    }

    fn linear(shape: Vec<usize>, value: f32) -> LinearWeight {
        LinearWeight::F32(filled(shape, value))
    }

    fn quantized_linear(shape: Vec<usize>, value: f32) -> LinearWeight {
        LinearWeight::Q8_0(quantize_q8_0(&filled(shape, value)).unwrap())
    }

    fn config() -> Qwen3Config {
        Qwen3Config {
            vocab_size: 16,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: Some(4),
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32,
            tie_word_embeddings: true,
            sliding_window: None,
            max_window_layers: None,
            use_sliding_window: false,
            rope_scaling: None,
            architectures: vec!["Qwen3ForCausalLM".to_owned()],
        }
    }

    fn layer() -> Qwen3LayerWeights {
        Qwen3LayerWeights {
            input_layernorm: filled(vec![8], 1.0),
            q_proj: linear(vec![8, 8], 0.01),
            q_norm: filled(vec![4], 1.0),
            k_proj: linear(vec![8, 4], 0.01),
            k_norm: filled(vec![4], 1.0),
            v_proj: linear(vec![8, 4], 0.01),
            o_proj: linear(vec![8, 8], 0.01),
            post_attention_layernorm: filled(vec![8], 1.0),
            gate_proj: linear(vec![8, 16], 0.01),
            up_proj: linear(vec![8, 16], 0.01),
            down_proj: linear(vec![16, 8], 0.01),
        }
    }

    fn quantized_layer() -> Qwen3LayerWeights {
        Qwen3LayerWeights {
            input_layernorm: filled(vec![8], 1.0),
            q_proj: quantized_linear(vec![8, 8], 0.01),
            q_norm: filled(vec![4], 1.0),
            k_proj: quantized_linear(vec![8, 4], 0.01),
            k_norm: filled(vec![4], 1.0),
            v_proj: quantized_linear(vec![8, 4], 0.01),
            o_proj: quantized_linear(vec![8, 8], 0.01),
            post_attention_layernorm: filled(vec![8], 1.0),
            gate_proj: quantized_linear(vec![8, 16], 0.01),
            up_proj: quantized_linear(vec![8, 16], 0.01),
            down_proj: quantized_linear(vec![16, 8], 0.01),
        }
    }

    #[test]
    fn runs_qwen3_prefill_and_decode() {
        let config = config();
        let model = Qwen3Model::new(
            config.clone(),
            CpuScalar::new(),
            filled(vec![config.vocab_size, config.hidden_size], 0.01),
            vec![layer()],
            filled(vec![config.hidden_size], 1.0),
            filled(vec![config.hidden_size, config.vocab_size], 0.01),
        )
        .unwrap();
        let mut cache = KvCache::new(
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_key_value_heads,
            config.effective_head_dim(),
        )
        .unwrap();
        let prefill = model.forward(&[1, 2], &mut cache).unwrap();
        assert_eq!(prefill.shape().dims(), &[2, config.vocab_size]);
        assert_eq!(cache.sequence_length(0).unwrap(), 2);
        let decode = model.forward(&[3], &mut cache).unwrap();
        assert_eq!(decode.shape().dims(), &[1, config.vocab_size]);
        assert_eq!(cache.sequence_length(0).unwrap(), 3);
    }

    #[test]
    fn runs_qwen3_with_q8_linear_weights() {
        let config = config();
        let model = Qwen3Model::new(
            config.clone(),
            CpuScalar::new(),
            filled(vec![config.vocab_size, config.hidden_size], 0.01),
            vec![quantized_layer()],
            filled(vec![config.hidden_size], 1.0),
            filled(vec![config.hidden_size, config.vocab_size], 0.01),
        )
        .unwrap();
        let mut cache = KvCache::new(
            config.num_hidden_layers,
            config.max_position_embeddings,
            config.num_key_value_heads,
            config.effective_head_dim(),
        )
        .unwrap();
        let logits = model.forward(&[1], &mut cache).unwrap();
        assert_eq!(logits.shape().dims(), &[1, config.vocab_size]);
    }
}
