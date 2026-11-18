//! Own each variant's temporary QIF artifacts so CI need not retain six copies.
use std::{
    collections::BTreeMap,
    env,
    ffi::OsStr,
    fs::{self, File},
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

use quantinfer_format::sha256_reader;

pub(super) struct Artifacts {
    pub single: PathBuf,
    pub package: PathBuf,
    owned: Option<PathBuf>,
}

impl Artifacts {
    pub fn open_or_publish(root: &Path, kind: &str, variant: &str) -> Self {
        let Some(converter) = env::var_os("QUANTINFER_TEST_QIF_PUBLISHER")
            .or_else(|| env::var_os("QUANTINFER_K_QUANT_CONVERTER"))
        else {
            return Self {
                single: root.join(format!("{kind}-{variant}.qif")),
                package: root.join(format!("{kind}-{variant}-package")),
                owned: None,
            };
        };
        Self::publish_input(&root.join(format!("{kind}-{variant}.gguf")), &converter)
    }

    fn publish_input(input: &Path, converter: &OsStr) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let owned = (0..100)
            .find_map(|_| {
                let path = env::temp_dir().join(format!(
                    "quantinfer-k-publication-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed)
                ));
                match fs::create_dir(&path) {
                    Ok(()) => Some(path),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                    Err(error) => panic!("create {}: {error}", path.display()),
                }
            })
            .expect("cannot reserve an unused publication directory");
        let artifacts = Self {
            single: owned.join("model.qif"),
            package: owned.join("package"),
            owned: Some(owned),
        };
        for (profile, codec, output) in [
            ("single", "none", &artifacts.single),
            ("package", "zstd", &artifacts.package),
        ] {
            let mut command = Command::new(converter);
            command
                .arg("from-gguf")
                .arg("--input")
                .arg(input)
                .arg("--output")
                .arg(output)
                .args(["--qif-profile", profile, "--qif-codec", codec]);
            if profile == "package" {
                command.args(["--max-shard-bytes", "16777216"]);
            }
            let result = command.output().expect("start QIF publisher");
            assert!(
                result.status.success(),
                "publisher failed: {} {}",
                String::from_utf8_lossy(&result.stdout),
                String::from_utf8_lossy(&result.stderr)
            );
        }
        artifacts
    }

    pub fn hashes(&self) -> BTreeMap<String, String> {
        let hash = |path: &Path| sha256_reader(&mut File::open(path).unwrap()).unwrap();
        let mut result = BTreeMap::from([("single/model.qif".to_owned(), hash(&self.single))]);
        for entry in fs::read_dir(&self.package).unwrap() {
            let entry = entry.unwrap();
            assert!(entry.file_type().unwrap().is_file());
            result.insert(
                format!("package/{}", entry.file_name().to_str().unwrap()),
                hash(&entry.path()),
            );
        }
        result
    }

    pub fn cleanup(mut self) {
        if let Some(path) = self.owned.take() {
            fs::remove_dir_all(path).expect("clean up owned QIF publication");
        }
    }
}

impl Drop for Artifacts {
    fn drop(&mut self) {
        if let Some(path) = self.owned.take() {
            // Preserve the original failure when the publisher or parity check panics.
            let _ = fs::remove_dir_all(path);
        }
    }
}
