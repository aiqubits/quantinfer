//! Validate optimization options before measurement or artifact publication.
use std::{
    fs,
    path::{Path, PathBuf},
};

use crate::common::required_path;

pub(crate) struct OptimizationOptions {
    pub policy: PathBuf,
    pub report: PathBuf,
}

fn destination(path: &Path) -> Result<PathBuf, String> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    Ok(fs::canonicalize(parent)
        .map_err(|e| e.to_string())?
        .join(path.file_name().ok_or("invalid output path")?))
}

impl OptimizationOptions {
    pub fn parse(
        arguments: &[String],
        output: &Path,
        manifest: Option<&Path>,
        ggml: bool,
    ) -> Result<Option<Self>, String> {
        let policy = arguments
            .iter()
            .any(|a| a == "--mixed-precision-optimize")
            .then(|| required_path(arguments, "--mixed-precision-optimize"))
            .transpose()?;
        let report = arguments
            .iter()
            .any(|a| a == "--mixed-precision-report")
            .then(|| required_path(arguments, "--mixed-precision-report"))
            .transpose()?;
        let (policy, report) =
            match (policy, report) {
                (None, None) => return Ok(None),
                (Some(policy), Some(report)) => (policy, report),
                _ => return Err(
                    "--mixed-precision-optimize requires --mixed-precision-report and vice versa"
                        .into(),
                ),
            };
        if ggml
            || arguments.iter().any(|a| {
                matches!(
                    a.as_str(),
                    "--mixed-precision"
                        | "--mixed-precision-policy"
                        | "--quantize"
                        | "--imatrix"
                        | "--ggml"
                )
            })
        {
            return Err("precision optimization requires native layout and owns its candidate/calibration policy; do not combine with other quantization flags".into());
        }
        match fs::symlink_metadata(&report) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => return Err("precision report must name a new file".into()),
            Err(e) => return Err(e.to_string()),
        }
        let report_path = destination(&report)?;
        let mut outputs = vec![destination(output)?];
        if let Some(manifest) = manifest {
            outputs.push(destination(manifest)?);
            let parent = manifest.parent().unwrap_or(Path::new("."));
            outputs.push(destination(&parent.join("checksums.sha256"))?);
        }
        if outputs.contains(&report_path) {
            return Err("precision report and artifact/manifest/checksum paths must differ".into());
        }
        Ok(Some(Self { policy, report }))
    }
}
