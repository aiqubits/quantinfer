use std::env;

fn main() {
    if cfg!(feature = "external") {
        println!("cargo:rerun-if-env-changed=QUANTINFER_GGML_LIB_DIR");
        let library_dir = env::var("QUANTINFER_GGML_LIB_DIR")
            .expect("set QUANTINFER_GGML_LIB_DIR to the pinned ggml-base library directory");
        println!("cargo:rustc-link-search=native={library_dir}");
        println!("cargo:rustc-link-lib=dylib=ggml-base");
    }
}
