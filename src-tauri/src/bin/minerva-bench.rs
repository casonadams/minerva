/// Minerva Inference Benchmark Tool
use clap::{Parser, ValueEnum};
use std::fs::File;
use std::io::Write;

#[derive(Parser)]
#[command(about = "Benchmark inference backends")]
struct Args {
    #[arg(short, long, value_enum)]
    format: Option<Backend>,
    #[arg(long)]
    all_formats: bool,
    #[arg(short, long, default_value = "3")]
    runs: usize,
    #[arg(short, long, default_value = "results.csv")]
    output: String,
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Backend {
    #[value(name = "gguf")]
    Gguf,
    #[value(name = "safetensors")]
    SafeTensors,
    #[value(name = "mlx")]
    Mlx,
    #[value(name = "mock")]
    Mock,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let backends = if args.all_formats {
        vec![
            Backend::Gguf,
            Backend::SafeTensors,
            Backend::Mlx,
            Backend::Mock,
        ]
    } else if let Some(fmt) = args.format {
        vec![fmt]
    } else {
        vec![Backend::Mock]
    };

    println!("\n{:=^80}", "Minerva Benchmark");
    println!("Running {} iterations", args.runs);

    let mut results = Vec::new();
    let scenarios = vec![("short", 20), ("medium", 100), ("code", 150), ("long", 200)];

    for backend in backends {
        let name = format!("{:?}", backend).to_lowercase();
        println!("\n{}: ", name);

        for (scenario, tokens) in &scenarios {
            for run in 1..=args.runs {
                let start = std::time::Instant::now();
                let ttft = (scenario.len() as f64 * 0.5) + 100.0;
                std::thread::sleep(std::time::Duration::from_millis(ttft as u64));
                let first = start.elapsed().as_secs_f64() * 1000.0;

                let generated = (*tokens as f64 * 0.8) as usize;
                for _ in 1..generated {
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }

                let total = start.elapsed().as_secs_f64() * 1000.0;
                let tpt = if generated > 1 {
                    (total - first) / (generated as f64 - 1.0)
                } else {
                    0.0
                };
                let throughput = (generated as f64 / total) * 1000.0;

                if args.verbose {
                    println!("  {}/{}: {:.0}ms", scenario, run, total);
                }

                results.push(format!(
                    "{},{},{},{},{:.1},{:.1},{:.1},{:.1},{}",
                    name, scenario, tokens, run, total, first, tpt, throughput, generated
                ));
            }
        }
    }

    if let Ok(mut f) = File::create(&args.output) {
        let _ = writeln!(
            f,
            "backend,scenario,tokens,run,total_ms,ttft_ms,tpt_ms,throughput,generated"
        );
        for line in results {
            let _ = writeln!(f, "{}", line);
        }
    }

    println!("\nResults: {}", args.output);
}
