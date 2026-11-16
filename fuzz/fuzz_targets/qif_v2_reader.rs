#![no_main]

use libfuzzer_sys::fuzz_target;
use quantinfer_format::qif::Reader as QifReader;

fuzz_target!(|data: &[u8]| {
    let _ = QifReader::open(data);
});
