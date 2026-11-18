#![no_main]
use libfuzzer_sys::fuzz_target;
use quantinfer_format_fuzz::{package_controls::PackageCases, MAX_INPUT};
use std::sync::OnceLock;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT || data.len() < 2 {
        return;
    }
    static CASES: OnceLock<PackageCases> = OnceLock::new();
    CASES.get_or_init(PackageCases::new).exercise(data);
});
