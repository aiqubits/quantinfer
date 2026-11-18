//! Generate all core encoding/layout/codec/profile fixtures, with tied weights.
use ed25519_dalek::SigningKey;
use quantinfer_core::{
    DType, QuantizationAxis, QuantizationScheme, Shape, TensorStorageDescriptor,
};
use quantinfer_format::{
    qif::{
        write_qif_package_with_codec, write_qif_signed, write_qif_with_codec, Codec, MetadataValue,
    },
    ModelManifest, TensorManifest,
};
use std::{
    collections::BTreeMap,
    env,
    fs::{self, File},
    path::Path,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let directory = env::args()
        .nth(1)
        .ok_or("usage: qif_profile_fixtures NEW_DIRECTORY")?;
    let root = Path::new(&directory);
    fs::create_dir(root)?;
    let mut model = ModelManifest::new("qif-core-profile-汉字🚀", "fixture");
    model.source_format = "synthetic".into();
    model.checksums = BTreeMap::from([
        (
            "a\0path".into(),
            quantinfer_format::sha256_hex(b"synthetic source bytes"),
        ),
        (
            "~qif-key:".into(),
            quantinfer_format::sha256_hex(b"other synthetic source bytes"),
        ),
        (
            "层/weight.bin".into(),
            quantinfer_format::sha256_hex(b"Unicode source name"),
        ),
    ]);
    model.quantization = QuantizationScheme::Mixed;
    // Unknown requirements are preserved by format readers; execution is a
    // separate policy layer and must reject unavailable mandatory requirements.
    model.compatibility.required_features = vec!["future.format.feature".into()];
    let mut payloads = BTreeMap::new();
    for (dtype, block, width, scheme) in [
        (DType::F32, 1, 4, QuantizationScheme::None),
        (DType::F16, 1, 2, QuantizationScheme::None),
        (DType::BF16, 1, 2, QuantizationScheme::None),
        (DType::Q4_0, 32, 18, QuantizationScheme::Q4_0),
        (DType::Q4_K, 256, 144, QuantizationScheme::Q4_K),
        (DType::Q5_K, 256, 176, QuantizationScheme::Q5_K),
        (DType::Q6_K, 256, 210, QuantizationScheme::Q6_K),
        (DType::Q8_0, 32, 34, QuantizationScheme::Q8_0),
    ] {
        for output_rows in [false, true] {
            let name = format!(
                "{dtype:?}-{}-\"\\\n",
                if output_rows { "output" } else { "row" }
            );
            let shape = Shape::from(if output_rows {
                vec![256, 2]
            } else {
                vec![2, 256]
            });
            let mut descriptor = if block == 1 {
                TensorStorageDescriptor::floating(dtype, shape)?
            } else {
                TensorStorageDescriptor::quantized(dtype, scheme, shape, block)?
            };
            if output_rows {
                descriptor.physical_layout = "ggml_output_rows".into();
                if block != 1 {
                    descriptor.quantization_axis = QuantizationAxis::Axis(0);
                }
            }
            descriptor.backend_requirements = vec!["future.backend".into()];
            let mut data = vec![0_u8; 512 / block * width];
            for (i, part) in data.chunks_exact_mut(width).enumerate() {
                if block == 1 {
                    let f = match i % 5 {
                        0 => 0.0,
                        1 => -0.0,
                        2 => -1.5,
                        3 => 0.125,
                        _ => 3.0,
                    };
                    match dtype {
                        DType::F32 => part.copy_from_slice(&f32::to_le_bytes(f)),
                        DType::F16 => part.copy_from_slice(&half::f16::from_f32(f).to_le_bytes()),
                        DType::BF16 => part.copy_from_slice(&half::bf16::from_f32(f).to_le_bytes()),
                        _ => unreachable!(),
                    }
                } else {
                    for (j, byte) in part.iter_mut().enumerate() {
                        *byte = (i * 17 + j * 37) as u8;
                    }
                    let offsets: &[usize] = match dtype {
                        DType::Q6_K => &[208],
                        DType::Q4_K | DType::Q5_K => &[0, 2],
                        _ => &[0],
                    };
                    for &offset in offsets {
                        part[offset..offset + 2].copy_from_slice(
                            &half::f16::from_f32(if i % 2 == 0 { 0.5 } else { -0.25 })
                                .to_le_bytes(),
                        );
                    }
                }
            }
            model.tensors.push(TensorManifest {
                name: name.clone(),
                descriptor: descriptor.clone(),
                checksum_sha256: None,
            });
            payloads.insert(name.clone(), data.clone());
            let alias = format!("{name}-alias");
            model.tensors.push(TensorManifest {
                name: alias.clone(),
                descriptor,
                checksum_sha256: None,
            });
            payloads.insert(alias, data);
        }
    }
    model.tensors.push(TensorManifest {
        name: "scalar\0tensor".into(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::scalar())?,
        checksum_sha256: None,
    });
    payloads.insert(
        "scalar\0tensor".into(),
        f32::MIN_POSITIVE.to_le_bytes().to_vec(),
    );
    model.tensors.push(TensorManifest {
        name: "scalar".into(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::scalar())?,
        checksum_sha256: None,
    });
    payloads.insert("scalar".into(), 0.000_001f32.to_le_bytes().to_vec());
    let metadata = BTreeMap::from([
        ("a.u32".into(), MetadataValue::U32(u32::MAX)),
        ("b.i32".into(), MetadataValue::I32(i32::MIN)),
        ("c.u64".into(), MetadataValue::U64(u64::MAX)),
        ("d.i64".into(), MetadataValue::I64(i64::MIN)),
        ("e.f32".into(), MetadataValue::F32(0.1)),
        ("f.f64".into(), MetadataValue::F64(-0.0)),
        ("g.bool".into(), MetadataValue::Bool(true)),
        (
            "h.string".into(),
            MetadataValue::String("汉字\n\"\\\0".into()),
        ),
        (
            "i.array".into(),
            MetadataValue::Array(vec![
                MetadataValue::String("one".into()),
                MetadataValue::String("two".into()),
            ]),
        ),
        ("j.empty".into(), MetadataValue::Array(vec![])),
        (
            "unicode.🚀".into(),
            MetadataValue::Map(BTreeMap::from([(
                "汉字".into(),
                MetadataValue::String("preserved".into()),
            )])),
        ),
        (
            "s.unicode".into(),
            MetadataValue::String("\u{10ffff}\u{2028}\u{2029}🚀".into()),
        ),
        ("n".into(), MetadataValue::String("ordinary".into())),
        (
            "n\0key".into(),
            MetadataValue::Map(BTreeMap::from([(
                "\0nested".into(),
                MetadataValue::Bool(false),
            )])),
        ),
        ("~qif-key:".into(), MetadataValue::String("prefix".into())),
        (
            "~qif-key:6162".into(),
            MetadataValue::String("hex-like suffix".into()),
        ),
        (
            "k.map".into(),
            MetadataValue::Map(BTreeMap::from([(
                "nested".into(),
                MetadataValue::Array(vec![MetadataValue::Map(BTreeMap::new())]),
            )])),
        ),
    ]);
    let key = SigningKey::from_bytes(&[42; 32]);
    fs::write(
        root.join("trusted-key.hex"),
        key.verifying_key()
            .to_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
    )?;
    for (name, codec) in [
        ("none", Codec::None),
        ("zstd", Codec::Zstd),
        ("lz4", Codec::Lz4),
    ] {
        let read = |name: &str| Ok(payloads[name].as_slice());
        write_qif_with_codec(
            model.clone(),
            &metadata,
            read,
            &mut File::create(root.join(format!("single-{name}.qif")))?,
            codec,
        )?;
        write_qif_signed(
            model.clone(),
            &metadata,
            read,
            &mut File::create(root.join(format!("signed-{name}.qif")))?,
            codec,
            &key,
        )?;
        write_qif_package_with_codec(
            model.clone(),
            &metadata,
            read,
            root.join(format!("package-{name}")),
            Some(192),
            codec,
        )?;
    }
    let mut empty = ModelManifest::new("empty-fixture", "fixture");
    empty.source_format = "synthetic".into();
    write_qif_with_codec(
        empty.clone(),
        &BTreeMap::new(),
        |_| {
            Err(quantinfer_format::FormatError::Invalid(
                "unexpected tensor".into(),
            ))
        },
        &mut File::create(root.join("empty.qif"))?,
        Codec::None,
    )?;
    write_qif_package_with_codec(
        empty,
        &BTreeMap::new(),
        |_| {
            Err(quantinfer_format::FormatError::Invalid(
                "unexpected tensor".into(),
            ))
        },
        root.join("empty-package"),
        None,
        Codec::None,
    )?;
    let mut large = ModelManifest::new("multi-chunk-fixture", "fixture");
    large.source_format = "synthetic".into();
    let elements = 1024 * 1024 + 3;
    large.tensors.push(TensorManifest {
        name: "weight".into(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![elements]))?,
        checksum_sha256: None,
    });
    let data = vec![0_u8; elements * 4];
    for (name, codec) in [("zstd", Codec::Zstd), ("lz4", Codec::Lz4)] {
        write_qif_with_codec(
            large.clone(),
            &BTreeMap::new(),
            |_| Ok(&data),
            &mut File::create(root.join(format!("multi-chunk-{name}.qif")))?,
            codec,
        )?;
    }
    for index in 0..16 {
        let mut alias = large.tensors[0].clone();
        alias.name = format!("alias-{index}");
        large.tensors.push(alias);
    }
    write_qif_with_codec(
        large,
        &BTreeMap::new(),
        |_| Ok(&data),
        &mut File::create(root.join("alias-report-budget.qif"))?,
        Codec::Zstd,
    )?;
    Ok(())
}
