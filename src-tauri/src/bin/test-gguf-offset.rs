use std::fs::File;
use std::io::{Read, Seek};

fn main() {
    let home = std::env::home_dir().unwrap();
    let path =
        home.join("Library/Caches/llama.cpp/ggml-org_gpt-oss-20b-GGUF_gpt-oss-20b-mxfp4.gguf");

    if !path.exists() {
        println!("GGUF file not found");
        return;
    }

    let mut file = File::open(&path).unwrap();
    let file_size = file.metadata().unwrap().len();
    println!(
        "File size: {} bytes ({:.1} MB)",
        file_size,
        file_size as f64 / 1_000_000.0
    );

    // Read first 512 bytes to understand structure
    let mut buf = [0u8; 512];
    file.read_exact(&mut buf).unwrap();

    println!("\nFirst 128 bytes (hex):");
    for i in 0..128 {
        if i % 16 == 0 {
            println!();
            print!("{:04x}: ", i);
        }
        print!("{:02x} ", buf[i]);
    }
    println!();
}
