/// Baseline performance measurement tool for Phase 4 Step 7
///
/// This binary performs comprehensive performance measurements for batch processing
/// operations without external dependencies. Measurements are saved to a report file.
use minerva_lib::inference::baseline_benchmarks;
use std::fs::File;
use std::io::Write;

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Phase 4 Step 7: Baseline Performance Measurement");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let mut report = String::new();
    report.push_str("# Baseline Performance Measurements - Phase 4 Step 7\n\n");
    report.push_str(&format!(
        "Generated: {}\n\n",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
    ));

    run_all_benchmarks(&mut report);
    write_report_file(&report);
}

fn run_all_benchmarks(report: &mut String) {
    println!("\n📊 TOKENIZATION BENCHMARKS\n");
    report.push_str(&baseline_benchmarks::run_tokenization_benchmarks());

    println!("\n📊 INFERENCE BENCHMARKS\n");
    report.push_str(&baseline_benchmarks::run_inference_benchmarks());

    println!("\n📊 STATISTICS CALCULATION BENCHMARKS\n");
    report.push_str(&baseline_benchmarks::run_statistics_benchmarks());

    println!("\n═══════════════════════════════════════════════════════════════════");
    println!("BASELINE MEASUREMENT COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════\n");

    report.push_str(&baseline_benchmarks::generate_summary());
}

fn write_report_file(report: &str) {
    let path = "/Users/cadams/src/github.com/casonadams/playground/docs/PHASE_4_STEP7_BASELINE_MEASUREMENTS.md";
    match File::create(path) {
        Ok(mut file) => {
            if let Err(e) = file.write_all(report.as_bytes()) {
                eprintln!("Error writing report: {}", e);
            } else {
                println!("✓ Baseline report written to: {}", path);
            }
        }
        Err(_) => eprintln!("Failed to create report file"),
    }
    println!("\nMeasurements complete. Check the report for detailed results.");
}
