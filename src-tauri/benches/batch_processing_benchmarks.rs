use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use minerva_lib::inference::batch::{
    BatchInferenceEngine, BatchItem, BatchTokenizer, InferenceBatchRequest, TokenizeBatchRequest,
};

// ============================================================================
// TOKENIZATION BENCHMARKS
// ============================================================================

fn bench_tokenizer_creation(c: &mut Criterion) {
    c.bench_function("tokenizer_creation", |b| b.iter(BatchTokenizer::new));
}

fn bench_single_text_tokenization(c: &mut Criterion) {
    let tokenizer = BatchTokenizer::new();
    let requests = vec![BatchItem::new(
        "req1".to_string(),
        TokenizeBatchRequest {
            text: black_box("Hello, world! This is a test of the batch tokenizer.".to_string()),
        },
    )];

    c.bench_function("tokenize_single_short_text", |b| {
        b.iter(|| tokenizer.encode_batch(black_box(requests.clone())))
    });
}

fn bench_batch_tokenization_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("tokenize_batch_sizes");

    for batch_size in [8, 16, 32, 64].iter() {
        let tokenizer = BatchTokenizer::new();
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(3);
        let requests: Vec<_> = (0..*batch_size)
            .map(|i| {
                BatchItem::new(
                    format!("req{}", i),
                    TokenizeBatchRequest { text: text.clone() },
                )
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| b.iter(|| tokenizer.encode_batch(black_box(requests.clone()))),
        );
    }
    group.finish();
}

fn bench_long_text_tokenization(c: &mut Criterion) {
    let tokenizer = BatchTokenizer::new();
    let long_text = "word ".repeat(1000); // ~5000 characters

    let requests = vec![BatchItem::new(
        "long_req".to_string(),
        TokenizeBatchRequest {
            text: black_box(long_text),
        },
    )];

    c.bench_function("tokenize_long_text_1000_words", |b| {
        b.iter(|| tokenizer.encode_batch(black_box(requests.clone())))
    });
}

fn bench_detokenization(c: &mut Criterion) {
    let mut group = c.benchmark_group("detokenize");
    let tokenizer = BatchTokenizer::new();

    for batch_size in [8, 32].iter() {
        let requests: Vec<_> = (0..*batch_size)
            .map(|i| {
                BatchItem::new(
                    format!("detok{}", i),
                    minerva_lib::inference::batch::DetokenizeBatchRequest {
                        tokens: vec![1, 2, 3, 4, 5],
                    },
                )
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| b.iter(|| tokenizer.decode_batch(black_box(requests.clone()))),
        );
    }
    group.finish();
}

// ============================================================================
// INFERENCE BENCHMARKS
// ============================================================================

fn bench_inference_engine_creation(c: &mut Criterion) {
    c.bench_function("inference_engine_creation", |b| {
        b.iter(BatchInferenceEngine::new)
    });
}

fn bench_single_inference(c: &mut Criterion) {
    let engine = BatchInferenceEngine::new();
    let requests = vec![BatchItem::new(
        "inf1".to_string(),
        InferenceBatchRequest {
            prompt: black_box("What is artificial intelligence?".to_string()),
            max_tokens: 100,
            temperature: 0.7,
        },
    )];

    c.bench_function("infer_single_prompt", |b| {
        b.iter(|| engine.infer_batch(black_box(requests.clone())))
    });
}

fn bench_batch_inference_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("infer_batch_sizes");

    for batch_size in [4, 8, 16].iter() {
        let engine = BatchInferenceEngine::new();
        let requests: Vec<_> = (0..*batch_size)
            .map(|i| {
                BatchItem::new(
                    format!("inf{}", i),
                    InferenceBatchRequest {
                        prompt: format!("Question {}?", i),
                        max_tokens: 50,
                        temperature: 0.5,
                    },
                )
            })
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| b.iter(|| engine.infer_batch(black_box(requests.clone()))),
        );
    }
    group.finish();
}

fn bench_temperature_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("infer_temperature");
    let engine = BatchInferenceEngine::new();

    for temp in [0.1, 0.5, 1.0, 1.5].iter() {
        let requests = vec![BatchItem::new(
            "temp_test".to_string(),
            InferenceBatchRequest {
                prompt: "Generate text".to_string(),
                max_tokens: 100,
                temperature: *temp,
            },
        )];

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.1}", temp)),
            temp,
            |b, _| b.iter(|| engine.infer_batch(black_box(requests.clone()))),
        );
    }
    group.finish();
}

fn bench_max_tokens_variations(c: &mut Criterion) {
    let mut group = c.benchmark_group("infer_max_tokens");
    let engine = BatchInferenceEngine::new();

    for max_tokens in [10, 50, 100, 200].iter() {
        let requests = vec![BatchItem::new(
            "tokens_test".to_string(),
            InferenceBatchRequest {
                prompt: "Generate text".to_string(),
                max_tokens: *max_tokens,
                temperature: 0.7,
            },
        )];

        group.bench_with_input(
            BenchmarkId::from_parameter(max_tokens),
            max_tokens,
            |b, _| b.iter(|| engine.infer_batch(black_box(requests.clone()))),
        );
    }
    group.finish();
}

// ============================================================================
// STATISTICS BENCHMARKS
// ============================================================================

fn bench_stats_calculation_various_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("stats_calculation");

    for size in [10, 100, 1000].iter() {
        let items = *size;
        let total_duration = items as u128 * 10; // ~10ms per item on average

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}items", items)),
            &items,
            |b, _| {
                b.iter(|| {
                    let _stats = minerva_lib::inference::batch::BatchStats::new(
                        black_box(*size),
                        black_box(total_duration),
                    );
                })
            },
        );
    }
    group.finish();
}

// ============================================================================
// COMPARISON BENCHMARKS: SINGLE VS BATCH
// ============================================================================

fn bench_single_vs_batch_tokenization(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_vs_batch_tokenization");

    let tokenizer = BatchTokenizer::new();
    let text = "The quick brown fox jumps over the lazy dog. ";

    // Single item
    let single_requests = vec![BatchItem::new(
        "single".to_string(),
        TokenizeBatchRequest {
            text: text.to_string(),
        },
    )];

    group.bench_function("single_item", |b| {
        b.iter(|| tokenizer.encode_batch(black_box(single_requests.clone())))
    });

    // Batch of 32
    let batch_requests: Vec<_> = (0..32)
        .map(|i| {
            BatchItem::new(
                format!("item{}", i),
                TokenizeBatchRequest {
                    text: text.to_string(),
                },
            )
        })
        .collect();

    group.bench_function("batch_32", |b| {
        b.iter(|| tokenizer.encode_batch(black_box(batch_requests.clone())))
    });

    group.finish();
}

fn bench_single_vs_batch_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_vs_batch_inference");

    let engine = BatchInferenceEngine::new();

    // Single item
    let single_requests = vec![BatchItem::new(
        "single".to_string(),
        InferenceBatchRequest {
            prompt: "Question?".to_string(),
            max_tokens: 50,
            temperature: 0.7,
        },
    )];

    group.bench_function("single_prompt", |b| {
        b.iter(|| engine.infer_batch(black_box(single_requests.clone())))
    });

    // Batch of 8
    let batch_requests: Vec<_> = (0..8)
        .map(|i| {
            BatchItem::new(
                format!("prompt{}", i),
                InferenceBatchRequest {
                    prompt: format!("Question {}?", i),
                    max_tokens: 50,
                    temperature: 0.7,
                },
            )
        })
        .collect();

    group.bench_function("batch_8", |b| {
        b.iter(|| engine.infer_batch(black_box(batch_requests.clone())))
    });

    group.finish();
}

// ============================================================================
// END-TO-END PIPELINE BENCHMARKS
// ============================================================================

fn bench_tokenize_then_infer(c: &mut Criterion) {
    let tokenizer = BatchTokenizer::new();
    let engine = BatchInferenceEngine::new();

    c.bench_function("e2e_tokenize_then_infer_batch8", |b| {
        b.iter(|| {
            // Tokenize 8 prompts
            let tok_requests: Vec<_> = (0..8)
                .map(|i| {
                    BatchItem::new(
                        format!("tok{}", i),
                        TokenizeBatchRequest {
                            text: format!("Prompt text {}", i),
                        },
                    )
                })
                .collect();

            let _tok_results = tokenizer.encode_batch(black_box(tok_requests));

            // Then run inference on 8 prompts
            let inf_requests: Vec<_> = (0..8)
                .map(|i| {
                    BatchItem::new(
                        format!("inf{}", i),
                        InferenceBatchRequest {
                            prompt: format!("Prompt text {}", i),
                            max_tokens: 100,
                            temperature: 0.7,
                        },
                    )
                })
                .collect();

            let _inf_results = engine.infer_batch(black_box(inf_requests));
        })
    });
}

// ============================================================================
// CRITERION GROUP DEFINITIONS
// ============================================================================

criterion_group!(
    tokenization_benches,
    bench_tokenizer_creation,
    bench_single_text_tokenization,
    bench_batch_tokenization_sizes,
    bench_long_text_tokenization,
    bench_detokenization,
);

criterion_group!(
    inference_benches,
    bench_inference_engine_creation,
    bench_single_inference,
    bench_batch_inference_sizes,
    bench_temperature_variations,
    bench_max_tokens_variations,
);

criterion_group!(stats_benches, bench_stats_calculation_various_sizes,);

criterion_group!(
    comparison_benches,
    bench_single_vs_batch_tokenization,
    bench_single_vs_batch_inference,
);

criterion_group!(e2e_benches, bench_tokenize_then_infer,);

criterion_main!(
    tokenization_benches,
    inference_benches,
    stats_benches,
    comparison_benches,
    e2e_benches
);
