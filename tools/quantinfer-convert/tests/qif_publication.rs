use quantinfer_format::{GgufReader, GgufValue, GgufWriter, PackedTensorReader};
use std::{fs, process::Command};

#[test]
fn publishes_both_qif_profiles_all_codecs_without_requantization_or_overwrite() {
    let root = std::env::temp_dir().join(format!("qif-publication-{}", std::process::id()));
    fs::create_dir(&root).unwrap();
    let input = root.join("input.gguf");
    let mut writer = GgufWriter::new();
    writer.insert_metadata("general.architecture", GgufValue::String("fixture".into()));
    writer
        .add_f32_tensor("weight", vec![2], &[1.0, 2.0])
        .unwrap();
    fs::write(&input, writer.finish().unwrap()).unwrap();
    let original = GgufReader::open(fs::read(&input).unwrap()).unwrap();
    for profile in ["single", "package"] {
        for codec in ["none", "zstd", "lz4"] {
            let output = root.join(format!("{profile}-{codec}"));
            let command = || {
                let mut cmd = Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"));
                cmd.arg("from-gguf")
                    .arg("--input")
                    .arg(&input)
                    .arg("--output")
                    .arg(&output)
                    .args(["--qif-profile", profile, "--qif-codec", codec]);
                if profile == "package" {
                    cmd.args(["--max-shard-bytes", "4096"]);
                }
                cmd
            };
            let result = command().output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            let published = PackedTensorReader::open_path(&output).unwrap();
            assert_eq!(published.cached_tensor_bytes(), 0);
            published.verify_qif().unwrap();
            assert_eq!(
                published.tensor_data("weight").unwrap().as_ref(),
                original.tensor_bytes("weight").unwrap()
            );
            assert!(!command().output().unwrap().status.success());
            PackedTensorReader::open_path(&output)
                .unwrap()
                .verify_qif()
                .unwrap();
        }
    }
    for options in [
        vec!["--qif-codec", "unknown"],
        vec!["--qif-profile", "unknown"],
        vec!["--max-shard-bytes", "4096"],
        vec!["--qif-profile", "package", "--max-shard-bytes", "0"],
    ] {
        let output = root.join("invalid");
        assert!(!Command::new(env!("CARGO_BIN_EXE_quantinfer-convert"))
            .arg("from-gguf")
            .arg("--input")
            .arg(&input)
            .arg("--output")
            .arg(&output)
            .args(options)
            .output()
            .unwrap()
            .status
            .success());
        assert!(!output.exists());
    }
    fs::remove_dir_all(root).unwrap();
}
