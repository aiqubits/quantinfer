//! Fixed-model native-layout K-quant quality and QIF publication matrix.
use quantinfer_format::GgmlType;
use serde_json::json;
use std::{env, fs, path::Path};
#[path = "support/dense_quality.rs"]
mod dense_quality;
#[path = "support/qif_publication.rs"]
mod qif_publication;
use dense_quality::{errors, probability_kl, run, top, Container, Fixture};
#[test]
fn probability_error_ignores_constant_logit_shifts() {
    let baseline = [1.0, 0.0, -1.0];
    let shifted = [21.0, 20.0, 19.0];
    assert!(errors(&baseline, &shifted).max > 10.0);
    assert!(probability_kl(&baseline, &shifted) < 1.0e-12);
    assert!(probability_kl(&[10.0, -10.0], &[-10.0, 10.0]) > 19.0);
    assert!(probability_kl(&[f32::MAX; 2], &[f32::MAX; 2]) < 1.0e-12);
}

#[test]
#[ignore = "requires verified SmolLM2/Qwen3 native-layout conversions"]
fn validates_native_k_quant_quality_and_qif_parity() {
    let kind = env::var("QUANTINFER_K_QUANT_MODEL").expect("set smollm2 or qwen3");
    let root = env::var_os("QUANTINFER_K_QUANT_DIR").expect("set verified artifact directory");
    let root = Path::new(&root);
    let fixture: Fixture = serde_json::from_str(match kind.as_str() {
        "smollm2" => include_str!("../../../tests/golden/smollm2-135m-f32.json"),
        "qwen3" => include_str!("../../../tests/golden/qwen3-0.6b-baseline.json"),
        _ => panic!("unsupported fixed model {kind}"),
    })
    .unwrap();
    let baseline = run(
        &root.join(format!("{kind}-f32.gguf")),
        &kind,
        &|_| GgmlType::F32,
        &fixture,
        Container::Gguf,
    );
    let baseline_top = top(&baseline.logits);
    let mut records = Vec::new();
    let mut violations = Vec::new();
    for (variant, dtype) in [
        ("q4_k", GgmlType::Q4_K),
        ("q5_k", GgmlType::Q5_K),
        ("q6_k", GgmlType::Q6_K),
    ] {
        let quantized = run(
            &root.join(format!("{kind}-{variant}.gguf")),
            &kind,
            &|_| dtype,
            &fixture,
            Container::Gguf,
        );
        let logits = errors(&baseline.logits, &quantized.logits);
        let weights = errors(&baseline.query, &quantized.query);
        let kl = probability_kl(&baseline.logits, &quantized.logits);
        let overlap = top(&quantized.logits)
            .iter()
            .filter(|id| baseline_top.contains(id))
            .count();
        // Bound next-token distribution divergence for this fixed prompt.
        // The separate legacy model tests retain their existing raw-logit
        // budgets. Raw logits also remain in this report: different layouts
        // and constant shifts do not imply the same probability error.
        if kl > 0.5 {
            violations.push(format!(
                "{kind} {variant} next-token KL {kl} exceeds 0.5 nats"
            ));
        }
        if overlap < 6 {
            violations.push(format!("{kind} {variant} top-10 overlap {overlap}"));
        }
        if kind == "qwen3" {
            if logits.max > 13.0 || logits.mse > 9.0 {
                violations.push(format!("{variant} logits {logits:?}"));
            }
            if weights.max > 0.032 || weights.mean > 0.0025 {
                violations.push(format!("{variant} weights {weights:?}"));
            }
        }
        let sampled = fixture.logits.as_ref().map(|values| {
            values
                .iter()
                .map(|&(id, value)| (f64::from(quantized.logits[id]) - f64::from(value)).abs())
                .fold(0.0, f64::max)
        });
        if kind == "smollm2" && weights.relative_l2 > 0.25 {
            violations.push(format!("{variant} weights {weights:?}"));
        }
        let mut record = json!({"model":kind,"variant":variant,"logits":logits,"sampled_transformers_max_error":sampled,"probability_kl_nats":kl,"query_weights":weights,"top_10_overlap":overlap,"generated":quantized.generated});
        if env::var_os("QUANTINFER_K_QUANT_QIF").is_some()
            || env::var_os("QUANTINFER_K_QUANT_CONVERTER").is_some()
        {
            let artifacts = qif_publication::Artifacts::open_or_publish(root, &kind, variant);
            for (path, profile, container) in [
                (&artifacts.single, "single_none", Container::SingleNone),
                (&artifacts.package, "package_zstd", Container::PackageZstd),
            ] {
                let qif = run(path, &kind, &|_| dtype, &fixture, container);
                assert_eq!(qif.inventory, quantized.inventory);
                assert_eq!(qif.query, quantized.query);
                assert_eq!(qif.logits, quantized.logits);
                assert_eq!(qif.generated, quantized.generated);
                println!("k_quant_qif_parity model={kind} variant={variant} profile={profile}");
            }
            record["qif_artifact_sha256"] = serde_json::to_value(artifacts.hashes()).unwrap();
            artifacts.cleanup();
        }
        println!("k_quant_quality={record}");
        records.push(record);
    }
    if let Some(path) = env::var_os("QUANTINFER_K_QUANT_REPORT") {
        fs::write(path,serde_json::to_vec_pretty(&json!({"source_commit":env::var("GITHUB_SHA").ok(),"model":kind,"records":records,"passed":violations.is_empty(),"violations":violations})).unwrap()).unwrap();
    }
    assert!(
        violations.is_empty(),
        "quality budget failures: {violations:?}"
    );
}
