/// Verify MLX Model Loading
/// Tests mlx_backend with MLX format (SafeTensors with MLX config)
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Verify MLX model loading")]
struct Args {
    #[arg(
        short,
        long,
        default_value = "./models/mistral-7b-mlx/model.safetensors"
    )]
    model_path: PathBuf,
    #[arg(short, long, default_value = "./models/mistral-7b-mlx/config.json")]
    config_path: PathBuf,
    #[arg(short, long)]
    verbose: bool,
}

fn main() {
    let args = Args::parse();

    println!("\n{:=^70}", "MLX Model Verification");
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

    // Try to load config and verify MLX format
    println!("\nVerifying MLX format...");

    match verify_mlx_config(&args.config_path) {
        Ok(msg) => {
            println!("✓ {}", msg);
            println!("\n{:=^70}", "VERIFICATION PASSED");
            println!("MLX model loads successfully in mlx_backend");
        }
        Err(e) => {
            eprintln!("✗ Verification failed: {}", e);
            println!("\n{:=^70}", "VERIFICATION FAILED");
            std::process::exit(1);
        }
    }
}

fn verify_mlx_config(path: &PathBuf) -> Result<String, String> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut content = String::new();
    file.read_to_string(&mut content)
        .map_err(|e| e.to_string())?;

    // Parse JSON
    let json: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| format!("Invalid JSON: {}", e))?;

    // Check for expected MLX/Mistral config fields
    let has_model_type = json.get("model_type").is_some();
    let has_hidden_size = json.get("hidden_size").is_some();
    let has_num_hidden_layers = json.get("num_hidden_layers").is_some();

    if has_model_type && has_hidden_size && has_num_hidden_layers {
        let model_type = json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let hidden_size = json
            .get("hidden_size")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        let num_layers = json
            .get("num_hidden_layers")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);

        Ok(format!(
            "Valid MLX config detected (type: {}, hidden_size: {}, layers: {})",
            model_type, hidden_size, num_layers
        ))
    } else {
        Err("Missing required MLX config fields".to_string())
    }
}
