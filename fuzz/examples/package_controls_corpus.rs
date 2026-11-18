fn main() {
    let output = std::env::args()
        .nth(1)
        .expect("usage: package_controls_corpus OUTPUT_DIR");
    quantinfer_format_fuzz::package_controls::PackageCases::new()
        .write_seeds(std::path::Path::new(&output));
}
