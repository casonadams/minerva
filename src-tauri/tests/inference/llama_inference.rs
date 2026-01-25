use minerva::inference::llama_inference::{
    Decoder, GenerationParams, SamplingParams, SamplingStrategy,
};

#[test]
fn test_decoder_creation() {
    let decoder = Decoder::new(32000, 2048);
    assert_eq!(decoder.vocab_size, 32000);
}

#[test]
fn test_decoder_sample_greedy() {
    let decoder = Decoder::new(100, 512);
    let logits = vec![0.1; 100];
    let token = decoder
        .sample_token(
            &logits,
            SamplingParams {
                temperature: 1.0,
                strategy: SamplingStrategy::Greedy,
            },
        )
        .unwrap();
    assert!(token < 100);
}

#[test]
fn test_decoder_sample_topk() {
    let decoder = Decoder::new(100, 512);
    let mut logits = vec![0.1; 100];
    logits[0] = 1.0;
    logits[1] = 0.8;

    let token = decoder
        .sample_token(
            &logits,
            SamplingParams {
                temperature: 1.0,
                strategy: SamplingStrategy::TopK(5),
            },
        )
        .unwrap();
    assert!(token < 100);
}

#[test]
fn test_decoder_sample_topp() {
    let decoder = Decoder::new(100, 512);
    let mut logits = vec![0.1; 100];
    logits[0] = 1.0;
    logits[1] = 0.9;

    let token = decoder
        .sample_token(
            &logits,
            SamplingParams {
                temperature: 1.0,
                strategy: SamplingStrategy::TopP(0.9),
            },
        )
        .unwrap();
    assert!(token < 100);
}

#[test]
fn test_decoder_invalid_temperature() {
    let decoder = Decoder::new(100, 512);
    let logits = vec![0.1; 100];
    let result = decoder.sample_token(
        &logits,
        SamplingParams {
            temperature: -1.0,
            strategy: SamplingStrategy::Greedy,
        },
    );
    assert!(result.is_err());
}

#[test]
fn test_decoder_invalid_topk() {
    let decoder = Decoder::new(100, 512);
    let logits = vec![0.1; 100];
    let result = decoder.sample_token(
        &logits,
        SamplingParams {
            temperature: 1.0,
            strategy: SamplingStrategy::TopK(0),
        },
    );
    assert!(result.is_err());
}

#[test]
fn test_decoder_invalid_topp() {
    let decoder = Decoder::new(100, 512);
    let logits = vec![0.1; 100];
    let result = decoder.sample_token(
        &logits,
        SamplingParams {
            temperature: 1.0,
            strategy: SamplingStrategy::TopP(0.0),
        },
    );
    assert!(result.is_err());
}

#[test]
fn test_sampling_strategy_greedy() {
    let strategy = SamplingStrategy::Greedy;
    match strategy {
        SamplingStrategy::Greedy => {} // Correct variant
        _ => unreachable!(),
    }
}

#[test]
fn test_sampling_strategy_topk() {
    let strategy = SamplingStrategy::TopK(10);
    match strategy {
        SamplingStrategy::TopK(k) => assert_eq!(k, 10),
        _ => unreachable!(),
    }
}

#[test]
fn test_sampling_strategy_topp() {
    let strategy = SamplingStrategy::TopP(0.9);
    match strategy {
        SamplingStrategy::TopP(p) => assert!((p - 0.9).abs() < 1e-5),
        _ => unreachable!(),
    }
}

#[test]
fn test_decoder_generate_sequence() {
    let decoder = Decoder::new(100, 512);
    let initial = vec![1];

    let result = decoder
        .generate(
            GenerationParams {
                initial_tokens: &initial,
                num_tokens: 5,
                sampling: SamplingParams {
                    temperature: 1.0,
                    strategy: SamplingStrategy::Greedy,
                },
            },
            |_tokens| Ok(vec![0.1; 100]),
        )
        .unwrap();

    assert_eq!(result.len(), 6); // initial + 5 generated
}

#[test]
fn test_decoder_invalid_logits_size() {
    let decoder = Decoder::new(100, 512);
    let logits = vec![0.1; 50]; // Wrong size
    let result = decoder.sample_token(
        &logits,
        SamplingParams {
            temperature: 1.0,
            strategy: SamplingStrategy::Greedy,
        },
    );
    assert!(result.is_err());
}

#[test]
fn test_decoder_empty_initial_tokens() {
    let decoder = Decoder::new(100, 512);
    let result = decoder.generate(
        GenerationParams {
            initial_tokens: &[],
            num_tokens: 5,
            sampling: SamplingParams {
                temperature: 1.0,
                strategy: SamplingStrategy::Greedy,
            },
        },
        |_tokens| Ok(vec![0.1; 100]),
    );
    assert!(result.is_err());
}

#[test]
fn test_decoder_sequence_too_long() {
    let decoder = Decoder::new(100, 512);
    let initial = vec![1; 400];
    let result = decoder.generate(
        GenerationParams {
            initial_tokens: &initial,
            num_tokens: 200,
            sampling: SamplingParams {
                temperature: 1.0,
                strategy: SamplingStrategy::Greedy,
            },
        },
        |_tokens| Ok(vec![0.1; 100]),
    );
    assert!(result.is_err());
}
