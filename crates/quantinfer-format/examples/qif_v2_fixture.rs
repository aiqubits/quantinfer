use std::collections::BTreeMap;
use std::env;
use std::fs::File;

use quantinfer_core::{DType, Shape, TensorStorageDescriptor};
use quantinfer_format::qif_v2::{write_qif_v2, QifV2MetadataValue};
use quantinfer_format::{ModelManifest, TensorManifest};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = env::args()
        .nth(1)
        .ok_or("usage: qif_v2_fixture OUTPUT.qif")?;
    let mut manifest = ModelManifest::new("qif-v2-fixture", "fixture");
    manifest.tensors.push(TensorManifest {
        name: "weight".to_owned(),
        descriptor: TensorStorageDescriptor::floating(DType::F32, Shape::from(vec![2]))?,
        checksum_sha256: None,
    });
    let metadata = BTreeMap::from([(
        "fixture.kind".to_owned(),
        QifV2MetadataValue::String("golden".to_owned()),
    )]);
    let payload = [0_u8, 0, 128, 63, 0, 0, 0, 64];
    let mut file = File::create(output)?;
    write_qif_v2(manifest, &metadata, |_| Ok(&payload), &mut file)?;
    Ok(())
}
