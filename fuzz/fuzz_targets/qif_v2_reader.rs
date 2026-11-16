#![no_main]

use libfuzzer_sys::fuzz_target;
use quantinfer_format::qif_v2::QifV2Reader;

fuzz_target!(|data: &[u8]| {
    let _ = QifV2Reader::open(data);
});
