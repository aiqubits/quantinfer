//! Fixed-prompt validation of a measured mixed-precision conversion and QIF parity.
use std::{collections::BTreeMap, env, fs, path::Path};

use quantinfer_format::{
    is_linear_projection_storage_name, sha256_reader, GgmlType, PackedTensorReader,
};
use serde_json::{json, Value};

#[path = "support/dense_quality.rs"]
mod dense_quality;
#[path = "support/qif_publication.rs"]
mod qif_publication;
use dense_quality::{errors, probability_kl, run, top, Container, Fixture};

#[test]
#[ignore = "requires verified F32 and measured precision artifacts"]
fn validates_optimized_precision_quality_and_publication() {
    let kind = env::var("QUANTINFER_OPTIMIZED_MODEL").expect("set smollm2 or qwen3");
    let root = env::var_os("QUANTINFER_OPTIMIZED_DIR").expect("set artifact directory");
    let root = Path::new(&root);
    let fixture: Fixture = serde_json::from_str(match kind.as_str() {
        "smollm2" => include_str!("../../../tests/golden/smollm2-135m-f32.json"),
        "qwen3" => include_str!("../../../tests/golden/qwen3-0.6b-baseline.json"),
        _ => panic!("unsupported fixed model {kind}"),
    })
    .unwrap();
    let report_path = root.join(format!("{kind}-precision-report.json"));
    let report: Value = serde_json::from_slice(&fs::read(&report_path).unwrap()).unwrap();
    let input = root.join(format!("{kind}-optimized.gguf"));
    let source_hash = sha256_reader(&mut fs::File::open(&input).unwrap()).unwrap();
    assert!(report["artifact_sha256"]
        .as_object()
        .unwrap()
        .values()
        .any(|v| v.as_str() == Some(&source_hash)));
    let mut planned = BTreeMap::new();
    for tensor in report["tensors"].as_array().unwrap() {
        let dtype = match tensor["selected"]["scheme"].as_str().unwrap() {
            "f32" => GgmlType::F32,
            "q8_0" => GgmlType::Q8_0,
            "q4_0" => GgmlType::Q4_0,
            "q4_k" => GgmlType::Q4_K,
            "q5_k" => GgmlType::Q5_K,
            "q6_k" => GgmlType::Q6_K,
            other => panic!("unrecognized selected encoding {other}"),
        };
        assert!(planned
            .insert(tensor["name"].as_str().unwrap().to_owned(), dtype)
            .is_none());
    }
    let baseline = run(
        &root.join(format!("{kind}-f32.gguf")),
        &kind,
        &|_| GgmlType::F32,
        &fixture,
        Container::Gguf,
    );
    if let Some(samples) = &fixture.logits {
        for &(id, expected) in samples {
            assert!((baseline.logits[id] - expected).abs() <= 0.001);
        }
    }
    let precision = |name: &str| *planned.get(name).expect("every projection is planned");
    let optimized = run(&input, &kind, &precision, &fixture, Container::Gguf);
    assert_eq!(
        planned.len(),
        optimized
            .inventory
            .keys()
            .filter(|name| is_linear_projection_storage_name(name))
            .count()
    );
    let logits = errors(&baseline.logits, &optimized.logits);
    let weights = errors(&baseline.query, &optimized.query);
    let kl = probability_kl(&baseline.logits, &optimized.logits);
    let overlap = top(&optimized.logits)
        .iter()
        .filter(|id| top(&baseline.logits).contains(id))
        .count();
    let encoded_bytes = PackedTensorReader::open_path(&input)
        .unwrap()
        .tensors()
        .iter()
        .map(|t| t.size as u64)
        .sum::<u64>();
    assert_eq!(
        encoded_bytes,
        report["selected_encoded_weight_bytes"].as_u64().unwrap()
    );
    assert!(encoded_bytes <= report["maximum_encoded_weight_bytes"].as_u64().unwrap());
    let artifacts = qif_publication::Artifacts::open_or_publish(root, &kind, "optimized");
    for (path, container) in [
        (&artifacts.single, Container::SingleNone),
        (&artifacts.package, Container::PackageZstd),
    ] {
        let qif = run(path, &kind, &precision, &fixture, container);
        assert_eq!(qif.inventory, optimized.inventory);
        assert_eq!(qif.query, optimized.query);
        assert_eq!(qif.logits, optimized.logits);
        assert_eq!(qif.generated, optimized.generated);
    }
    let passed = kl <= 0.5 && overlap >= 6 && weights.relative_l2 <= 0.25;
    let quality = json!({"model":kind,"source_gguf_sha256":source_hash,"policy_sha256":report["policy_sha256"],
        "encoded_weight_bytes":encoded_bytes,"maximum_encoded_weight_bytes":report["maximum_encoded_weight_bytes"],
        "logits":logits,"query_weights":weights,"probability_kl_nats":kl,"top_10_overlap":overlap,
        "generated":optimized.generated,"qif_artifact_sha256":artifacts.hashes(),"passed":passed});
    println!("optimized_precision_quality={quality}");
    if let Some(path) = env::var_os("QUANTINFER_OPTIMIZED_QUALITY_REPORT") {
        fs::write(path, serde_json::to_vec_pretty(&quality).unwrap()).unwrap();
    }
    artifacts.cleanup();
    assert!(
        passed,
        "optimized precision exceeds the fixed-prompt quality budget"
    );
}
