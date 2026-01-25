use clap::{Parser, Subcommand};
use minerva_lib::cli;

#[derive(Parser)]
#[command(
    name = "Minerva",
    about = "Local LLM Server with OpenAI-compatible API",
    version = "0.1.0",
    author = "Cason Adams"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the API server
    Serve(cli::ServeArgs),
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Serve(args) => {
            if let Err(e) = cli::serve_command(args).await {
                eprintln!("Error: {}", e);
                std::process::exit(1);
            }
        }
    }
}
