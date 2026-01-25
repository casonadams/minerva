use crate::inference::batch::{
    BatchInferenceEngine, BatchItem, BatchTokenizer, InferenceBatchRequest, TokenizeBatchRequest,
};
use crate::inference::batch_measurement::measure_operation;

/// Run tokenization benchmarks
pub fn run_tokenization_benchmarks() -> String {
    let mut report = String::from("## Tokenization Benchmarks\n\n");

    let tokenizer = BatchTokenizer::new();
    let short_text = "Hello, world!";

    let short_stats = measure_operation("Tokenize single short text", 100, || {
        let requests = vec![BatchItem::new(
            "test".to_string(),
            TokenizeBatchRequest {
                text: short_text.to_string(),
            },
        )];
        let _ = tokenizer.encode_batch(requests);
    });
    report.push_str(&format!("- Single short text: {}\n", short_stats));

    let batch8_text = "The quick brown fox jumps over the lazy dog. ";
    let batch8_stats = measure_operation("Tokenize batch of 8", 50, || {
        let requests: Vec<_> = (0..8)
            .map(|i| {
                BatchItem::new(
                    format!("req{}", i),
                    TokenizeBatchRequest {
                        text: batch8_text.to_string(),
                    },
                )
            })
            .collect();
        let _ = tokenizer.encode_batch(requests);
    });
    report.push_str(&format!("- Batch of 8: {}\n", batch8_stats));

    let batch32_stats = measure_operation("Tokenize batch of 32", 20, || {
        let requests: Vec<_> = (0..32)
            .map(|i| {
                BatchItem::new(
                    format!("req{}", i),
                    TokenizeBatchRequest {
                        text: batch8_text.to_string(),
                    },
                )
            })
            .collect();
        let _ = tokenizer.encode_batch(requests);
    });
    report.push_str(&format!("- Batch of 32: {}\n", batch32_stats));

    report.push_str("\nSpeedup (single vs batch):\n");
    report.push_str("(See performance analysis for calculated speedups)\n");

    let long_text = "word ".repeat(100);
    let long_stats = measure_operation("Tokenize long text (500 chars)", 50, || {
        let requests = vec![BatchItem::new(
            "long".to_string(),
            TokenizeBatchRequest {
                text: long_text.to_string(),
            },
        )];
        let _ = tokenizer.encode_batch(requests);
    });
    report.push_str(&format!("\n- Long text (500 chars): {}\n", long_stats));

    report.push_str("\n## Detokenization Benchmarks\n\n");

    let detok_stats = measure_operation("Detokenize single batch", 100, || {
        let requests = vec![BatchItem::new(
            "detok".to_string(),
            crate::inference::batch::DetokenizeBatchRequest {
                tokens: vec![1, 2, 3, 4, 5],
            },
        )];
        let _ = tokenizer.decode_batch(requests);
    });
    report.push_str(&format!("- Single batch: {}\n", detok_stats));

    report
}

/// Run inference benchmarks
pub fn run_inference_benchmarks() -> String {
    let mut report = String::from("\n## Inference Benchmarks\n\n");

    let engine = BatchInferenceEngine::new();

    let single_inf_stats = measure_operation("Infer single prompt", 50, || {
        let requests = vec![BatchItem::new(
            "inf".to_string(),
            InferenceBatchRequest {
                prompt: "What is AI?".to_string(),
                max_tokens: 100,
                temperature: 0.7,
            },
        )];
        let _ = engine.infer_batch(requests);
    });
    report.push_str(&format!("- Single prompt: {}\n", single_inf_stats));

    let batch4_inf_stats = measure_operation("Infer batch of 4", 25, || {
        let requests: Vec<_> = (0..4)
            .map(|i| {
                BatchItem::new(
                    format!("inf{}", i),
                    InferenceBatchRequest {
                        prompt: format!("Question {}?", i),
                        max_tokens: 100,
                        temperature: 0.7,
                    },
                )
            })
            .collect();
        let _ = engine.infer_batch(requests);
    });
    report.push_str(&format!("- Batch of 4: {}\n", batch4_inf_stats));

    let batch8_inf_stats = measure_operation("Infer batch of 8", 15, || {
        let requests: Vec<_> = (0..8)
            .map(|i| {
                BatchItem::new(
                    format!("inf{}", i),
                    InferenceBatchRequest {
                        prompt: format!("Question {}?", i),
                        max_tokens: 100,
                        temperature: 0.7,
                    },
                )
            })
            .collect();
        let _ = engine.infer_batch(requests);
    });
    report.push_str(&format!("- Batch of 8: {}\n", batch8_inf_stats));

    report.push_str("\nSpeedup (single vs batch):\n");
    report.push_str("(See performance analysis for calculated speedups)\n");

    run_temperature_variations(&engine, &mut report);

    report
}

/// Run temperature variation benchmarks
fn run_temperature_variations(engine: &BatchInferenceEngine, report: &mut String) {
    report.push_str("\n## Temperature Variations\n\n");

    for temp in [0.1, 0.5, 1.0, 1.5] {
        let temp_stats = measure_operation(&format!("Infer with temp={:.1}", temp), 25, || {
            let requests = vec![BatchItem::new(
                "temp".to_string(),
                InferenceBatchRequest {
                    prompt: "Generate".to_string(),
                    max_tokens: 100,
                    temperature: temp,
                },
            )];
            let _ = engine.infer_batch(requests);
        });
        report.push_str(&format!("- Temperature {:.1}: {}\n", temp, temp_stats));
    }
}

/// Run statistics calculation benchmarks
pub fn run_statistics_benchmarks() -> String {
    let mut report = String::from("\n## Statistics Calculation\n\n");

    for size in [10, 100, 1000] {
        let stats_name = format!("Calculate stats for {} items", size);
        let stats_bench = measure_operation(&stats_name, 100, || {
            let _ = crate::inference::batch::BatchStats::new(size, size as u128 * 10);
        });
        report.push_str(&format!("- {} items: {}\n", size, stats_bench));
    }

    report
}

/// Generate performance summary section
pub fn generate_summary() -> String {
    let mut report = String::from("\n## Performance Summary\n\n");
    report
        .push_str("These baseline measurements establish the starting point for Phase 4 Step 7.\n");
    report.push_str("Future optimizations will compare against these baselines.\n\n");
    report.push_str("### Key Metrics to Track\n");
    report.push_str("- Tokenization speedup: Target 5-8x for batch 32\n");
    report.push_str("- Inference speedup: Target 3-5x for batch 8\n");
    report.push_str("- Statistics calculation: Target < 1ms for 100 items\n");
    report.push_str("- Memory overhead: Target < 5%\n\n");

    report.push_str("## Next Steps\n");
    report.push_str("1. Profile with flamegraph to identify hot paths\n");
    report.push_str("2. Analyze memory usage with heaptrack\n");
    report.push_str("3. Identify optimization opportunities\n");
    report.push_str("4. Implement optimizations incrementally\n");
    report.push_str("5. Re-measure and compare against baselines\n");

    report
}
