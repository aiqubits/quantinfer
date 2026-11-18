use std::collections::BTreeMap;
use std::env;
use std::fs::File;

use ed25519_dalek::SigningKey;
use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::qif::{write_qif_signed, write_qif_with_codec, Codec, MetadataValue};
use quantinfer_format::{ModelManifest, TensorManifest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = env::args()
        .nth(1)
        .ok_or("usage: qif_v2_fixture OUTPUT.qif [none|zstd|lz4] [signed]")?;
    let codec = match env::args().nth(2).as_deref() {
        None | Some("none") => Codec::None,
        Some("zstd") => Codec::Zstd,
        Some("lz4") => Codec::Lz4,
        Some(_) => return Err("codec must be none, zstd or lz4".into()),
    };
    let signed = match env::args().nth(3).as_deref() {
        None => false,
        Some("signed") => true,
        Some(_) => return Err("expected optional signed profile".into()),
    };
    if env::args().nth(4).is_some() {
        return Err("unexpected arguments".into());
    }
    let mut manifest = ModelManifest::new("qif-v2-fixture", "fixture");
    manifest.tensors.push(TensorManifest {
        name: "weight".to_owned(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2]))?,
        checksum_sha256: None,
    });
    let metadata = BTreeMap::from([(
        "fixture.kind".to_owned(),
        MetadataValue::String("golden".to_owned()),
    )]);
    let payload = [0_u8, 0, 128, 63, 0, 0, 0, 64];
    let mut file = File::create(output)?;
    if signed {
        write_qif_signed(
            manifest,
            &metadata,
            |_| Ok(&payload),
            &mut file,
            codec,
            &SigningKey::from_bytes(&[42; 32]),
        )?;
    } else {
        write_qif_with_codec(manifest, &metadata, |_| Ok(&payload), &mut file, codec)?;
    }
    Ok(())
}
