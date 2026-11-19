use quantinfer::{
    cpu::{load_gguf_tokenizer, CpuModel, ModelSource},
    model::TensorRole,
    runtime::{generate_stream, DecoderModel, GenerationConfig, StreamControl},
    tokenizer::Tokenizer,
};
use quantinfer_adapter_gptq::{GptqProvider, HessianCalibration};
use quantinfer_backend_cpu::CpuScalar;
use quantinfer_core::{QuantizationScheme, Tensor};
use quantinfer_format::{qif, sha256_hex, GgufReader, GgufWriter};
use quantinfer_kernel::KernelBackend;
use quantinfer_quant::{CalibratedQuantizationProvider, QuantizationProvider, QuantizedData};
use serde_json::{json, Value};
use std::{
    fs,
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

struct OwnedDirectory(PathBuf);
impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn mse(actual: &Tensor, expected: &Tensor) -> f64 {
    actual
        .as_f32()
        .unwrap()
        .iter()
        .zip(expected.as_f32().unwrap())
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>()
        / actual.as_f32().unwrap().len() as f64
}

fn quality(actual: &[f32], expected: &[f32]) -> (f64, usize) {
    let log_sum = |values: &[f32]| {
        let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        maximum
            + values
                .iter()
                .map(|&x| (f64::from(x) - maximum).exp())
                .sum::<f64>()
                .ln()
    };
    let (a, b) = (log_sum(actual), log_sum(expected));
    let kl = actual
        .iter()
        .zip(expected)
        .map(|(&x, &y)| {
            let reference = f64::from(y) - b;
            reference.exp() * (reference - (f64::from(x) - a))
        })
        .sum::<f64>();
    let top = |values: &[f32]| {
        let mut indices: Vec<_> = (0..values.len()).collect();
        indices.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        indices.truncate(10);
        indices
    };
    let left = top(actual);
    let right = top(expected);
    (kl, left.iter().filter(|i| right.contains(i)).count())
}

#[test]
#[ignore = "requires a verified native SmolLM2 F32 GGUF and embedded tokenizer"]
fn calibrated_real_projection_preserves_native_qif_model_quality() {
    let source_path =
        PathBuf::from(std::env::var_os("QUANTINFER_GPTQ_SOURCE_GGUF").expect("verified source"));
    let source_bytes = fs::read(source_path).unwrap();
    let source_hash = sha256_hex(&source_bytes);
    assert_eq!(
        source_hash,
        std::env::var("QUANTINFER_GPTQ_SOURCE_SHA256").expect("source hash")
    );
    let source = GgufReader::open(source_bytes).unwrap();
    assert!(!source.uses_ggml_output_rows().unwrap());
    let reference = CpuModel::from_gguf(&source, None, None).unwrap();
    let spec = CpuModel::spec_from_gguf(&source, None).unwrap();
    assert_eq!((spec.hidden_size, spec.layers.len()), (576, 30));
    let tokenizer = load_gguf_tokenizer(&source).unwrap();
    let prompts = [
        "Hello from the CPU runtime.",
        "A short example of model quantization.",
        "Numbers and words have different patterns.",
    ];
    let mut calibration = HessianCalibration::new(576, 576 * 576 * 8).unwrap();
    let mut rows = Vec::new();
    let mut observer = |layer, roles: &[TensorRole], activations: &Tensor| {
        if layer == 0 && roles.contains(&TensorRole::QueryProjection) {
            calibration.observe_rows(activations).unwrap();
            rows.extend_from_slice(activations.as_f32().unwrap());
        }
        Ok(())
    };
    for prompt in prompts {
        let tokens = tokenizer.encode(prompt).unwrap();
        reference
            .observe_projections(
                &tokens,
                &mut reference.new_cache(64).unwrap(),
                &mut observer,
            )
            .unwrap();
    }
    let activation_rows = Tensor::from_f32(vec![calibration.rows(), 576], rows).unwrap();
    let name = "blk.0.attn_q.weight";
    let weights = source.tensor_f32(name).unwrap();
    let provider = GptqProvider::default();
    let quantized = provider
        .quantize_calibrated(&weights, &calibration)
        .unwrap();
    let restored = provider.dequantize(&quantized).unwrap();
    let baseline_projection = CpuScalar::new().matmul(&activation_rows, &weights).unwrap();
    let projection = CpuScalar::new()
        .matmul(&activation_rows, &restored)
        .unwrap();
    let projection_mse = mse(&projection, &baseline_projection);
    let QuantizedData::Q4_0(quantized) = quantized else {
        unreachable!()
    };
    let mut rtn = vec![0.; 576 * 576];
    for input in 0..576 {
        for output in 0..576 {
            let scale = quantized.blocks()[output * 18 + input / 32].scale;
            let value = f64::from(weights.as_f32().unwrap()[input * 576 + output]);
            let code = (value / f64::from(scale)).round_ties_even().clamp(-8., 7.);
            rtn[input * 576 + output] = (f64::from(scale) * code) as f32;
        }
    }
    let rtn_projection = CpuScalar::new()
        .matmul(
            &activation_rows,
            &Tensor::from_f32(vec![576, 576], rtn).unwrap(),
        )
        .unwrap();
    let rtn_mse = mse(&rtn_projection, &baseline_projection);
    assert!(
        projection_mse < rtn_mse,
        "GPTQ {projection_mse} vs RTN {rtn_mse}"
    );

    let golden: Value =
        serde_json::from_str(include_str!("../../../tests/golden/smollm2-135m-f32.json")).unwrap();
    let prompt: Vec<usize> = golden["prompt_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let config = GenerationConfig {
        max_new_tokens: 8,
        eos_token_id: Some(0),
        ..Default::default()
    };
    let expected = reference
        .forward(&prompt, &mut reference.new_cache(64).unwrap())
        .unwrap();
    let vocabulary = spec.vocabulary.size;
    let expected_last = &expected.as_f32().unwrap()[(prompt.len() - 1) * vocabulary..];
    for pair in golden["logits"].as_array().unwrap() {
        assert!(
            (f64::from(expected_last[pair[0].as_u64().unwrap() as usize])
                - pair[1].as_f64().unwrap())
            .abs()
                < 0.001
        );
    }
    let baseline_tokens = generate_stream(
        &reference,
        &prompt,
        &mut reference.new_cache(64).unwrap(),
        &config,
        &mut |_| Ok(StreamControl::Continue),
    )
    .unwrap();
    let golden_tokens: Vec<usize> = golden["generated_tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(baseline_tokens, golden_tokens);
    drop(reference);

    let owned = OwnedDirectory(std::env::temp_dir().join(format!(
            "quantinfer-gptq-model-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )));
    fs::create_dir(&owned.0).unwrap();
    let mut intermediate = Vec::new();
    qif::import_gguf(&source, "smollm2-calibrated-gptq-query", &mut intermediate).unwrap();
    let native = qif::Reader::open(&intermediate).unwrap();
    let mut manifest = native.manifest().clone();
    manifest.quantization = QuantizationScheme::Mixed;
    for variant in &mut manifest.execution_variants {
        variant.quantization = QuantizationScheme::Mixed;
    }
    let tensor = manifest
        .tensors
        .iter_mut()
        .find(|t| t.name == name)
        .unwrap();
    tensor.descriptor = provider
        .storage_descriptor(weights.shape().clone())
        .unwrap();
    tensor.checksum_sha256 = None;
    let mut packed = GgufWriter::new();
    packed.add_q4_0_tensor(name, &quantized);
    let packed = GgufReader::open(packed.finish().unwrap()).unwrap();
    let get = |tensor: &str| {
        if tensor == name {
            packed.tensor_bytes(name)
        } else {
            native.tensor_bytes(tensor)
        }
    };
    let mut metadata = native.metadata().clone();
    metadata.insert(
        "quantinfer.quantization.algorithm".into(),
        qif::MetadataValue::String("gptq-static-signed-f16".into()),
    );
    let single = owned.0.join("model.qif");
    qif::write_qif(
        manifest.clone(),
        &metadata,
        get,
        &mut fs::File::create(&single).unwrap(),
    )
    .unwrap();
    let package = owned.0.join("package");
    qif::write_qif_package_with_codec(
        manifest,
        &metadata,
        get,
        &package,
        Some(64 * 1024 * 1024),
        qif::Codec::Zstd,
    )
    .unwrap();
    drop(native);
    drop(intermediate);
    let mut reports = Vec::new();
    let mut first_logits = None;
    for (profile, path) in [("single/none", single), ("package/zstd", package)] {
        let reader = ModelSource::open(path).unwrap().open_reader(None).unwrap();
        reader.verify_qif().unwrap();
        assert_eq!(
            reader.tensor_q4_0(name).unwrap().blocks(),
            quantized.blocks()
        );
        let model = CpuModel::from_gguf(&reader, None, None).unwrap();
        let actual = model
            .forward(&prompt, &mut model.new_cache(64).unwrap())
            .unwrap();
        if let Some(previous) = &first_logits {
            assert_eq!(&actual, previous);
        } else {
            first_logits = Some(actual.clone());
        }
        assert!(actual.as_f32().unwrap().iter().all(|v| v.is_finite()));
        let (kl, top10) = quality(
            &actual.as_f32().unwrap()[(prompt.len() - 1) * vocabulary..],
            expected_last,
        );
        assert!(
            (-1e-9..0.05).contains(&kl) && top10 >= 8,
            "KL {kl}, top10 {top10}"
        );
        let mut cache = model.new_cache(64).unwrap();
        let tokens = generate_stream(&model, &prompt, &mut cache, &config, &mut |_| {
            Ok(StreamControl::Continue)
        })
        .unwrap();
        let repeated = generate_stream(
            &model,
            &prompt,
            &mut model.new_cache(64).unwrap(),
            &config,
            &mut |_| Ok(StreamControl::Continue),
        )
        .unwrap();
        assert_eq!(tokens, repeated);
        for layer in 0..30 {
            assert_eq!(
                cache.sequence_length(layer).unwrap(),
                prompt.len() + tokens.len()
                    - usize::from(tokens.last().copied() == config.eos_token_id)
            );
        }
        assert_eq!(reader.cached_tensor_bytes(), 0);
        if let Some(previous) = reports.first() {
            let previous: &Value = previous;
            assert_eq!(previous["generated"], json!(tokens));
        }
        reports.push(json!({"profile":profile,"kl":kl,"top10_overlap":top10,"generated":tokens,"query_raw_sha256":sha256_hex(reader.tensor_data(name).unwrap().as_ref())}));
    }
    let report = json!({"source_sha256":source_hash,"quantized_tensor":name,"calibration_prompts":prompts,"calibration_rows":calibration.rows(),"projection_mse":projection_mse,"same_grid_rtn_mse":rtn_mse,"baseline_generated":baseline_tokens,"profiles":reports,"passed":true});
    println!("gptq_model_quality={report}");
    if let Some(path) = std::env::var_os("QUANTINFER_GPTQ_REPORT") {
        fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}
