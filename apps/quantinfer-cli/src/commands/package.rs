//! Model package inspection and profile loading commands.

use std::collections::BTreeMap;

use quantinfer_format::ModelPackage;
use quantinfer_runtime::InferenceProfile;

use super::common::{has_flag, option_value};

pub(crate) fn open_package(arguments: &[String]) -> Result<ModelPackage, String> {
    let path = option_value(arguments, "--input")?
        .ok_or_else(|| "provide a model package directory with --input".to_owned())?;
    ModelPackage::open(path).map_err(|error| error.to_string())
}

pub(crate) fn verify_package(arguments: &[String]) -> Result<(), String> {
    let package = open_package(arguments)?;
    let verified = package.verify_all().map_err(|error| error.to_string())?;
    if has_flag(arguments, "--json") {
        println!(
            "{}",
            serde_json::json!({ "manifest": package.manifest, "verified_artifacts": verified })
        );
    } else {
        println!("model_id={}", package.manifest.model_id);
        println!("architecture={}", package.manifest.architecture);
        println!("quantization={}", package.manifest.quantization.name());
        println!("verified_artifacts={verified}");
    }
    Ok(())
}

pub(crate) fn list_profiles(arguments: &[String]) -> Result<(), String> {
    let package = open_package(arguments)?;
    let profiles = package
        .manifest
        .checksums
        .keys()
        .filter_map(|path| path.strip_prefix("profiles/")?.strip_suffix(".json"))
        .map(|name| {
            let profile = load_inference_profile(Some(&package), name)?;
            profile
                .declared_variant(&package.manifest)
                .map_err(|error| format!("invalid profile {name}: {error}"))?;
            Ok((name, profile))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    if has_flag(arguments, "--json") {
        println!(
            "{}",
            serde_json::to_string(&profiles).map_err(|error| error.to_string())?
        );
    } else {
        for (name, profile) in profiles {
            println!(
                "{name} variant={} quantization={} context={} threads={}",
                profile.model_variant,
                profile.quantization_variant,
                profile.context_length,
                profile.thread_count
            );
        }
    }
    Ok(())
}

pub(crate) fn load_inference_profile(
    package: Option<&ModelPackage>,
    name: &str,
) -> Result<InferenceProfile, String> {
    let package =
        package.ok_or_else(|| "--inference-profile requires a model package".to_owned())?;
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(
            "inference profile names may contain letters, digits, hyphens, and underscores"
                .to_owned(),
        );
    }
    let bytes = package
        .read_verified(&format!("profiles/{name}.json"))
        .map_err(|error| format!("cannot verify inference profile {name}: {error}"))?;
    let profile: InferenceProfile = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse inference profile {name}: {error}"))?;
    profile.validate().map_err(|error| error.to_string())?;
    Ok(profile)
}
