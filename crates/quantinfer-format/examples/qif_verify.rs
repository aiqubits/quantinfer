//! Verify a SINGLE or PACKAGE artifact through the file-backed format API.
use quantinfer_format::qif::{FileReader, PackageReader};
use std::{env, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = env::args().nth(1).ok_or("usage: qif_verify INPUT")?;
    if Path::new(&path).is_dir() {
        PackageReader::open(path)?.verify()?;
    } else {
        FileReader::open(path)?.verify()?;
    }
    println!("QIF verified");
    Ok(())
}
