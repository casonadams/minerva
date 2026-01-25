/// Mistral 7B Downloader from HuggingFace Hub
use clap::{Parser, ValueEnum};
use indicatif::ProgressBar;
use std::{fs, path::PathBuf};
use tokio;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Format {
    #[value(name = "gguf")]
    Gguf,
    #[value(name = "safetensors")]
    SafeTensors,
    #[value(name = "mlx")]
    Mlx,
}

#[derive(Parser)]
#[command(about = "Download Mistral 7B")]
struct Args {
    #[arg(short, long)]
    model: Option<String>,
    #[arg(short, long, value_enum)]
    format: Option<Format>,
    #[arg(long)]
    all: bool,
    #[arg(short, long, default_value = "./models")]
    output: String,
    #[arg(short = 'f', long)]
    hf_token: Option<String>,
    #[arg(short = 'F', long)]
    files: Option<String>,
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let token = args
        .hf_token
        .clone()
        .or_else(|| std::env::var("HF_TOKEN").ok());

    if let Some(model_id) = &args.model {
        let files: Vec<_> = args
            .files
            .as_ref()
            .map(|l| l.split(',').map(|s| s.trim()).collect())
            .unwrap_or_else(|| vec!["model.safetensors", "config.json"]);

        for file in files {
            let url = format!("https://huggingface.co/{}/resolve/main/{}", model_id, file);
            println!("{}", file);
            let _ = download(&url, file, &token, args.verbose).await;
        }
        return;
    }

    let models = if args.all {
        vec![
            (
                "gguf",
                "TheBloke/Mistral-7B-GGUF",
                vec!["Mistral-7B.Q4_K_M.gguf"],
            ),
            (
                "safetensors",
                "mistralai/Mistral-7B",
                vec!["model.safetensors", "config.json", "tokenizer.json"],
            ),
            (
                "mlx",
                "mlx-community/Mistral-7B",
                vec!["model.safetensors", "config.json", "tokenizer.json"],
            ),
        ]
    } else if let Some(fmt) = args.format {
        let (name, repo, files) = match fmt {
            Format::Gguf => (
                "gguf",
                "TheBloke/Mistral-7B-GGUF",
                vec!["Mistral-7B.Q4_K_M.gguf"],
            ),
            Format::SafeTensors => (
                "safetensors",
                "mistralai/Mistral-7B",
                vec!["model.safetensors", "config.json", "tokenizer.json"],
            ),
            Format::Mlx => (
                "mlx",
                "mlx-community/Mistral-7B",
                vec!["model.safetensors", "config.json", "tokenizer.json"],
            ),
        };
        vec![(name, repo, files)]
    } else {
        return;
    };

    for (name, repo, files) in models {
        let path = PathBuf::from(&args.output).join(name);
        let _ = fs::create_dir_all(&path);
        for file in files {
            let url = format!("https://huggingface.co/{}/resolve/main/{}", repo, file);
            let _ = download(&url, file, &token, args.verbose).await;
        }
    }
}

async fn download(
    url: &str,
    name: &str,
    token: &Option<String>,
    verbose: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} ... ", name);

    let mut req = reqwest::Client::new().get(url);
    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {}", t));
    }

    let res = req.send().await?;
    let total = res.content_length().unwrap_or(0);
    let pb = if verbose {
        ProgressBar::new(total)
    } else {
        ProgressBar::hidden()
    };

    let mut stream = res.bytes_stream();
    use futures::stream::StreamExt;

    while let Some(Ok(chunk)) = stream.next().await {
        pb.inc(chunk.len() as u64);
    }

    pb.finish_and_clear();
    println!("✓");
    Ok(())
}
