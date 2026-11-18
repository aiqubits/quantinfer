fn main() {
    println!("cargo:rerun-if-changed=vendor-revision");
    println!("cargo:rerun-if-env-changed=QUANTINFER_KLEIDIAI_SOURCE_DIR");
    #[cfg(feature = "native")]
    build_native();
}

#[cfg(feature = "native")]
fn build_native() {
    use std::{env, path::PathBuf, process::Command};
    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
        || env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux")
        || env::var("CARGO_CFG_TARGET_ENDIAN").as_deref() != Ok("little")
    {
        return;
    }
    let source = PathBuf::from(env::var_os("QUANTINFER_KLEIDIAI_SOURCE_DIR").expect(
        "native AArch64 KleidiAI requires QUANTINFER_KLEIDIAI_SOURCE_DIR at the pinned checkout",
    ));
    let revision = Command::new("git")
        .arg("-C")
        .arg(&source)
        .args(["rev-parse", "HEAD"])
        .output()
        .expect("cannot inspect KleidiAI revision");
    assert!(
        revision.status.success(),
        "KleidiAI source must be a Git checkout"
    );
    assert_eq!(
        String::from_utf8(revision.stdout).unwrap().trim(),
        include_str!("vendor-revision").trim(),
        "KleidiAI revision does not match the reviewed ABI"
    );
    let clean = Command::new("git")
        .arg("-C")
        .arg(&source)
        .args(["status", "--porcelain"])
        .output()
        .expect("cannot inspect KleidiAI worktree");
    assert!(
        clean.status.success() && clean.stdout.is_empty(),
        "KleidiAI source checkout must be clean"
    );
    let family = "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod";
    let files = [
        format!("{family}.c"),
        format!("{family}_asm.S"),
        "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c".into(),
        "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c".into(),
    ];
    let mut build = cc::Build::new();
    build
        .include(&source)
        .flag("-fvisibility=hidden")
        .flag("-march=armv8.2-a+dotprod")
        .flag_if_supported("-std=c11")
        .opt_level(3)
        .file("src/native/bridge.c");
    println!("cargo:rerun-if-changed=src/native/bridge.c");
    for file in files {
        let path = source.join(file);
        if path.extension().is_some_and(|extension| extension == "c") {
            // Namespace this pinned SDK copy, including its assembly entry,
            // so another optional adapter's KleidiAI cannot interpose a
            // different packed-layout ABI in the same process/static link.
            for line in std::fs::read_to_string(&path)
                .expect("cannot read reviewed KleidiAI source")
                .lines()
                .map(str::trim_start)
            {
                if line.starts_with("size_t kai_") || line.starts_with("void kai_") {
                    let name = line
                        .split('(')
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .last()
                        .unwrap();
                    build.define(name, format!("quantinfer_kleidiai_{name}").as_str());
                }
            }
        }
        println!("cargo:rerun-if-changed={}", path.display());
        build.file(path);
    }
    build.compile("quantinfer_kleidiai");
    println!("cargo:rustc-link-lib=m");
}
