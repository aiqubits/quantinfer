#![no_main]

use libfuzzer_sys::fuzz_target;
use quantinfer_format_fuzz::{exercise_single, repair_single_root, MAX_INPUT};

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT {
        return;
    }
    exercise_single(data);
    if let Some(repaired) = repair_single_root(data) {
        exercise_single(&repaired);
    }
});
