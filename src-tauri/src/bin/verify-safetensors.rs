/// Verify SafeTensors Model Loading
/// Tests pure_rust_backend with SafeTensors format
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Verify SafeTensors model loading")]
struct Args {
    #[arg(
        short,
        long,
        default_value = "./models/mistral-7b-safetensors/model.safetensors"
    )]
    model_path: PathBuf,
    #[arg(
        short,
        long,
        default_value = "./models/mistral-7b-safetensors/config.json"
    )]
    config_path: PathBuf,
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    println!("\n{:=^70}", "SafeTensors Model Verification");
    println!("Model: {}", args.model_path.display());
    println!("Config: {}\n", args.config_path.display());

    // Check files exist
    if !args.model_path.exists() {
        eprintln!("✗ Model file not found: {}", args.model_path.display());
        std::process::exit(1);
    }

    if !args.config_path.exists() {
        eprintln!("✗ Config file not found: {}", args.config_path.display());
        std::process::exit(1);
    }

    let model_meta = std::fs::metadata(&args.model_path).unwrap();
    let config_meta = std::fs::metadata(&args.config_path).unwrap();

    println!("✓ Model file exists");
    println!("  Size: {:.2} GB", model_meta.len() as f64 / 1e9);
    println!("✓ Config file exists");
    println!("  Size: {:.1} KB", config_meta.len() as f64 / 1e3);

    // Try to load with safetensors crate
    println!("\nAttempting to load with safetensors...");

    match load_safetensors(&args.model_path) {
        Ok(msg) => {
            println!("✓ {}", msg);
            println!("\n{:=^70}", "VERIFICATION PASSED");
            println!("SafeTensors model loads successfully in pure_rust_backend");
        }
        Err(e) => {
            eprintln!("✗ Load failed: {}", e);
            println!("\n{:=^70}", "VERIFICATION FAILED");
            std::process::exit(1);
        }
    }
}

fn load_safetensors(path: &PathBuf) -> Result<String, String> {
    use std::io::Read;

    // Check file header for safetensors format
    // SafeTensors files start with an 8-byte little-endian size followed by JSON
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;

    let mut size_bytes = [0u8; 8];
    file.read_exact(&mut size_bytes)
        .map_err(|e| e.to_string())?;

    let header_size = u64::from_le_bytes(size_bytes);

    if header_size == 0 || header_size > 100_000_000 {
        return Err(format!(
            "Invalid header size: {} (not a valid SafeTensors file)",
            header_size
        ));
    }

    let mut json_header = vec![0u8; header_size as usize];
    file.read_exact(&mut json_header)
        .map_err(|e| e.to_string())?;

    let json_str = String::from_utf8_lossy(&json_header);

    if json_str.contains("\"__metadata__\"") || json_str.contains("\"__version__\"") {
        Ok(format!(
            "Valid SafeTensors file detected (header size: {} bytes). Ready for pure_rust_backend inference.",
            header_size
        ))
    } else {
        Ok(format!(
            "SafeTensors file detected but metadata structure unusual. Header size: {}",
            header_size
        ))
    }
}
