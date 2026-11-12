use std::env;

fn main() {
    if cfg!(feature = "external") {
        println!("cargo:rerun-if-env-changed=QUANTINFER_GGML_LIB_DIR");
        let library_dir = match env::var("QUANTINFER_GGML_LIB_DIR") {
            Ok(path) if !path.is_empty() => path,
            _ => {
                eprintln!(
                    "quantinfer-adapter-ggml: feature `external` requires \
                     QUANTINFER_GGML_LIB_DIR pointing to the pinned ggml-base library"
                );
                std::process::exit(1);
            }
        };
        if !std::path::Path::new(&library_dir).is_dir() {
            eprintln!(
                "quantinfer-adapter-ggml: QUANTINFER_GGML_LIB_DIR is not a directory: {library_dir}"
            );
            std::process::exit(1);
        }
        println!("cargo:rustc-link-search=native={library_dir}");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
    }
}
