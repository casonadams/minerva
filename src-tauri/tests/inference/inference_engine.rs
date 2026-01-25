use minerva::inference::engine_config::InferenceEngineConfig;
use minerva::inference::inference_engine::InferenceEngine;
use minerva::inference::model_weights::{LayerWeights, ModelWeights};

fn create_dummy_weights(config: &InferenceEngineConfig) -> ModelWeights {
    let mut layers = Vec::new();

    for _ in 0..config.num_layers {
        layers.push(LayerWeights {
            attn_norm_scale: vec![1.0; config.hidden_size],
            ffn_norm_scale: vec![1.0; config.hidden_size],
            ff_up: vec![0.1; config.hidden_size * config.intermediate_size],
            ff_down: vec![0.1; config.intermediate_size * config.hidden_size],
        });
    }

    ModelWeights {
        embeddings: vec![0.1; config.vocab_size * config.hidden_size],
        layers,
        final_norm_scale: vec![1.0; config.hidden_size],
        output_proj: vec![0.01; config.hidden_size * config.vocab_size],
    }
}

#[test]
fn test_engine_creation_valid() {
    let config = InferenceEngineConfig::tiny(100);
    let weights = create_dummy_weights(&config);
    let engine = InferenceEngine::new(config, weights);
    assert!(engine.is_ok());
}

#[test]
fn test_engine_creation_invalid_embeddings() {
    let mut config = InferenceEngineConfig::tiny(100);
    let mut weights = create_dummy_weights(&config);

    // Wrong embedding shape
    weights.embeddings = vec![0.1; 50];
    config.vocab_size = 100;

    let result = InferenceEngine::new(config, weights);
    assert!(result.is_err());
}

#[test]
fn test_forward_pass_shapes() {
    let config = InferenceEngineConfig::tiny(100);
    let weights = create_dummy_weights(&config);
    let engine = InferenceEngine::new(config, weights).unwrap();

    let tokens = vec![1, 2, 3];
    let logits = engine.forward(&tokens).unwrap();

    // Should have shape (seq_len, vocab_size)
    assert_eq!(logits.len(), 3 * 100);
}

#[test]
fn test_forward_pass_empty_sequence() {
    let config = InferenceEngineConfig::tiny(100);
    let weights = create_dummy_weights(&config);
    let engine = InferenceEngine::new(config, weights).unwrap();

    let result = engine.forward(&[]);
    assert!(result.is_err());
}

#[test]
fn test_forward_pass_sequence_too_long() {
    let config = InferenceEngineConfig::tiny(100);
    let weights = create_dummy_weights(&config);
    let engine = InferenceEngine::new(config, weights).unwrap();

    let tokens = vec![1; engine.config().max_seq_len + 1];
    let result = engine.forward(&tokens);
    assert!(result.is_err());
}

#[test]
fn test_forward_with_softmax_shapes() {
    let config = InferenceEngineConfig::tiny(100);
    let weights = create_dummy_weights(&config);
    let engine = InferenceEngine::new(config, weights).unwrap();

    let tokens = vec![1, 2, 3];
    let probs = engine.forward_with_softmax(&tokens).unwrap();

    assert_eq!(probs.len(), 3 * 100);
}

#[test]
fn test_softmax_probabilities_sum_to_one() {
    let config = InferenceEngineConfig::tiny(100);
    let weights = create_dummy_weights(&config);
    let engine = InferenceEngine::new(config, weights).unwrap();

    let tokens = vec![1, 2];
    let probs = engine.forward_with_softmax(&tokens).unwrap();

    // Check that each position sums to approximately 1.0
    for pos in 0..2 {
        let start = pos * 100;
        let end = start + 100;
        let sum: f32 = probs[start..end].iter().sum();
        assert!((sum - 1.0).abs() < 0.001, "Position {} sum: {}", pos, sum);
    }
}

#[test]
fn test_forward_numerical_stability() {
    let config = InferenceEngineConfig::tiny(100);
    let weights = create_dummy_weights(&config);
    let engine = InferenceEngine::new(config, weights).unwrap();

    let tokens = vec![1, 2, 3];
    let logits = engine.forward(&tokens).unwrap();

    // All values should be finite
    for &val in &logits {
        assert!(val.is_finite(), "Found non-finite value: {}", val);
    }
}

#[test]
fn test_forward_single_token() {
    let config = InferenceEngineConfig::tiny(50);
    let weights = create_dummy_weights(&config);
    let engine = InferenceEngine::new(config, weights).unwrap();

    let logits = engine.forward(&[1]).unwrap();
    assert_eq!(logits.len(), 50);
}

#[test]
fn test_config_consistency() {
    let config = InferenceEngineConfig::tiny(1000);
    let weights = create_dummy_weights(&config);
    let engine = InferenceEngine::new(config.clone(), weights).unwrap();

    assert_eq!(engine.config().vocab_size, 1000);
    assert_eq!(engine.config().hidden_size, 64);
    assert_eq!(engine.config().num_heads, 2);
}
