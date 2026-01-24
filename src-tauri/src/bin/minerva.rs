use std::env;
use minerva_lib::cli;

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        print_help();
        return;
    }
    
    match args[1].as_str() {
        "serve" => {
            match cli::serve::ServeArgs::from_args(&args[2..]) {
                Ok(serve_args) => {
                    if let Err(e) = cli::serve_command(serve_args).await {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    }
                }
                Err(e) => {
                    eprintln!("Error parsing arguments: {}", e);
                    std::process::exit(1);
                }
            }
        }
        "--help" | "-h" => {
            print_help();
        }
        "--version" | "-v" => {
            println!("Minerva v0.1.0");
        }
        cmd => {
            eprintln!("Unknown command: {}", cmd);
            print_help();
            std::process::exit(1);
        }
    }
}

fn print_help() {
    println!("Minerva - Local LLM Server with OpenAI-compatible API");
    println!();
    println!("USAGE:");
    println!("    minerva <COMMAND> [OPTIONS]");
    println!();
    println!("COMMANDS:");
    println!("    serve       Start the API server");
    println!("    --help      Print help information");
    println!("    --version   Print version information");
    println!();
    println!("OPTIONS for serve:");
    println!("    --host <HOST>             Server host (default: 127.0.0.1)");
    println!("    --port <PORT>             Server port (default: 3000)");
    println!("    --models-dir <PATH>       Models directory");
    println!("    --config <PATH>           Configuration file");
    println!("    --workers <COUNT>         Worker threads");
}
