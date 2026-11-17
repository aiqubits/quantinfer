//! Generate larger fixed fixtures for the bounded-memory verification check.
use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::{
    qif::{write_qif_package_with_codec, write_qif_with_codec, Codec},
    ModelManifest, TensorManifest,
};
use std::{collections::BTreeMap, env, fs, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(
        env::args()
            .nth(1)
            .ok_or("usage: qif_resource_fixture NEW_DIRECTORY")?,
    );
    fs::create_dir(&root)?;
    let mut payload = vec![0; 64 * 1024 * 1024];
    for (index, value) in payload.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *value = ((index % 1021) as f32).to_le_bytes();
    }
    let mut model = ModelManifest::new("resource-fixture", "fixture");
    model.tensors.push(TensorManifest {
        name: "weight".into(),
        descriptor: TensorStorageDescriptor::floating(
            DType::F32,
            Shape::from(vec![payload.len() / 4]),
        )?,
        checksum_sha256: None,
    });
    model.tensors.push(TensorManifest {
        name: "alias".into(),
        ..model.tensors[0].clone()
    });
    for (name, codec) in [
        ("none", Codec::None),
        ("zstd", Codec::Zstd),
        ("lz4", Codec::Lz4),
    ] {
        write_qif_with_codec(
            model.clone(),
            &BTreeMap::new(),
            |_| Ok(&payload),
            &mut fs::File::create(root.join(format!("{name}.qif")))?,
            codec,
        )?;
        write_qif_package_with_codec(
            model.clone(),
            &BTreeMap::new(),
            |_| Ok(&payload),
            root.join(format!("package-{name}")),
            Some(1024 * 1024),
            codec,
        )?;
    }
    let mut names = ModelManifest::new("name-fixture", "fixture");
    for id in 0..100 {
        names.tensors.push(TensorManifest {
            name: format!("t{id}"),
            descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2]))?,
            checksum_sha256: None,
        });
    }
    write_qif_with_codec(
        names,
        &BTreeMap::new(),
        |_| Ok(&[0; 8]),
        &mut fs::File::create(root.join("name-seed.qif"))?,
        Codec::None,
    )?;
    Ok(())
}
