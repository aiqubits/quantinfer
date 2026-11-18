//! Verify a SINGLE or PACKAGE artifact through the file-backed format API.
use quantinfer_format::qif::{FileReader, PackageReader};
use std::{env, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut arguments = env::args().skip(1);
    let first = arguments
        .next()
        .ok_or("usage: qif_verify [--packed] INPUT")?;
    let packed = first == "--packed";
    let path = if packed {
        arguments.next().ok_or("missing INPUT")?
    } else {
        first
    };
    if packed {
        let reader = quantinfer_format::PackedTensorReader::open_path(path)?;
        reader.verify_qif()?;
        assert_eq!(reader.cached_tensor_bytes(), 0);
    } else if Path::new(&path).is_dir() {
        PackageReader::open(path)?.verify()?;
    } else {
        FileReader::open(path)?.verify()?;
    }
    println!("QIF verified");
    Ok(())
}
