// GPU Inference Test Binary
// Tests basic model loading and forward pass with TinyLlama-1.1B SafeTensors

use std::fs;
use std::path::Path;
use std::time::Instant;

fn main() {
    println!("\n=== GPU SafeTensors Inference Test ===\n");

    let model_path = Path::new("../models/tinyllama-1.1b-safetensors/model.safetensors");
    let config_path = Path::new("../models/tinyllama-1.1b-safetensors/config.json");

    // Check files exist
    if !model_path.exists() {
        eprintln!("Error: Model file not found: {}", model_path.display());
        std::process::exit(1);
    }
    if !config_path.exists() {
        eprintln!("Error: Config file not found: {}", config_path.display());
        std::process::exit(1);
    }

    println!("✓ Model files found");

    // Get file sizes
    let model_size = fs::metadata(model_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  - Model: {} ({:.2} GB)",
        model_path.display(),
        model_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    let config_size = fs::metadata(config_path).map(|m| m.len()).unwrap_or(0);
    println!(
        "  - Config: {} ({} bytes)",
        config_path.display(),
        config_size
    );
    println!();

    // Read and parse config
    match fs::read_to_string(config_path) {
        Ok(content) => match serde_json::from_str::<serde_json::Value>(&content) {
            Ok(config) => {
                println!("✓ Config loaded successfully");

                if let Some(hidden_size) = config.get("hidden_size").and_then(|v| v.as_u64()) {
                    println!("  - Hidden size: {}", hidden_size);
                }
                if let Some(num_layers) = config.get("num_hidden_layers").and_then(|v| v.as_u64()) {
                    println!("  - Num layers: {}", num_layers);
                }
                if let Some(vocab_size) = config.get("vocab_size").and_then(|v| v.as_u64()) {
                    println!("  - Vocab size: {}", vocab_size);
                }
                if let Some(archs) = config.get("architectures").and_then(|v| v.as_array()) {
                    println!(
                        "  - Architectures: {:?}",
                        archs.iter().filter_map(|a| a.as_str()).collect::<Vec<_>>()
                    );
                }
            }
            Err(e) => {
                eprintln!("Error parsing config: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            eprintln!("Error reading config: {}", e);
            std::process::exit(1);
        }
    }

    println!();
    println!("To run full inference test:");
    println!("  cd src-tauri");
    println!("  cargo test --lib inference::gpu::backend::tests::test_backend_creation -- --ignored --nocapture 2>&1");
    println!();
    println!("This will load the actual SafeTensors model and measure forward pass timing.");
}
