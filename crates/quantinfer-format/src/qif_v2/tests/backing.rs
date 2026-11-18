use super::*;
use crate::PackedTensorReader;
use std::{
    fs,
    io::{Seek, SeekFrom, Write},
};

#[test]
fn lazy_runtime_storage_preserves_aliases_metadata_and_rejects_cold_corruption() {
    let first = [0, 0, 128, 63, 0, 0, 0, 64];
    let second = [0, 0, 64, 64, 0, 0, 128, 64];
    let metadata = BTreeMap::from([(
        "optional.hints".into(),
        MetadataValue::Map(BTreeMap::from([(
            "prefetch".into(),
            MetadataValue::Bool(true),
        )])),
    )]);
    for codec in [Codec::None, Codec::Zstd, Codec::Lz4] {
        for package in [false, true] {
            let path = std::env::temp_dir().join(format!(
                "qif-lazy-{}-{}-{package}",
                std::process::id(),
                codec.id()
            ));
            let mut manifest = sample_manifest();
            for name in ["cold", "alias"] {
                manifest.tensors.push(TensorManifest {
                    name: name.into(),
                    ..manifest.tensors[0].clone()
                });
            }
            let get = |name: &str| -> crate::Result<&[u8]> {
                Ok(if name == "cold" { &second } else { &first })
            };
            if package {
                write_qif_v2_package_with_codec(manifest, &metadata, get, &path, Some(4096), codec)
                    .unwrap();
            } else {
                write_qif_v2_with_codec(
                    manifest,
                    &metadata,
                    get,
                    &mut fs::File::create(&path).unwrap(),
                    codec,
                )
                .unwrap();
            }
            let reader = PackedTensorReader::open_path(&path).unwrap();
            assert!(reader.is_file_backed());
            assert_eq!(reader.cached_tensor_bytes(), 0);
            reader.verify_qif().unwrap();
            assert_eq!(reader.cached_tensor_bytes(), 0);
            assert_eq!(reader.qif_metadata().unwrap(), &metadata);
            let limit = reader.file_backing.as_ref().unwrap().control_length + 16;
            assert!(
                PackedTensorReader::open_path_with_memory_budget(&path, Some(limit - 1)).is_err()
            );
            PackedTensorReader::open_path_with_memory_budget(&path, Some(limit)).unwrap();
            // Damage only the unrequested physical tensor, retaining valid controls.
            let (payload, offset) = if package {
                let native = PackageReader::open(&path).unwrap();
                (
                    path.join(native.tensor_shard("cold").unwrap()),
                    native.tensor_info("cold").unwrap().data_offset,
                )
            } else {
                let bytes = fs::read(&path).unwrap();
                let native = Reader::open(&bytes).unwrap();
                (
                    path.clone(),
                    u64::from_le_bytes(bytes[72..80].try_into().unwrap())
                        + native.tensor_info("cold").unwrap().data_offset,
                )
            };
            let mut file = fs::OpenOptions::new().write(true).open(payload).unwrap();
            file.seek(SeekFrom::Start(offset)).unwrap();
            file.write_all(&[0xff]).unwrap();
            drop(file);
            let lazy = PackedTensorReader::open_path(&path).unwrap();
            assert_eq!(
                lazy.tensor_f32("weight").unwrap().as_f32().unwrap(),
                [1.0, 2.0]
            );
            assert_eq!(lazy.cached_tensor_bytes(), 0);
            assert!(lazy.tensor_data("cold").is_err());
            assert!(lazy.verify_qif().is_err());
            let clone = lazy.clone();
            assert_eq!(
                lazy.tensor_bytes("alias").unwrap().as_ptr(),
                clone.tensor_bytes("weight").unwrap().as_ptr()
            );
            assert_eq!(lazy.cached_tensor_bytes(), first.len());
            assert_eq!(clone.cached_tensor_bytes(), first.len());
            drop((reader, lazy, clone));
            if package {
                fs::remove_dir_all(path).unwrap();
            } else {
                fs::remove_file(path).unwrap();
            }
        }
    }
}
