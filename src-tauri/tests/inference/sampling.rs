use minerva_lib::inference::greedy_sampling::sample_greedy;
use minerva_lib::inference::softmax_utils::softmax;
use minerva_lib::inference::temperature::{apply_temperature, TemperatureConfig};
use minerva_lib::inference::temperature_sampling::sample_temperature;
use minerva_lib::inference::top_k_sampling::{sample_top_k, TopKConfig};
use minerva_lib::inference::top_p_sampling::{sample_top_p, TopPConfig};

#[test]
fn test_temperature_config_creation() {
    let standard = TemperatureConfig::standard();
    assert_eq!(standard.temperature, 1.0);

    let sharp = TemperatureConfig::sharp();
    assert!(sharp.temperature < 1.0);

    let soft = TemperatureConfig::soft();
    assert!(soft.temperature > 1.0);
}

#[test]
fn test_apply_temperature_unchanged() {
    let logits = vec![1.0, 2.0, 3.0];
    let scaled = apply_temperature(&logits, 1.0);
    assert_eq!(scaled, logits);
}

#[test]
fn test_apply_temperature_scaling() {
    let logits = vec![1.0, 2.0, 3.0];

    // Temperature < 1 should amplify differences
    let sharp = apply_temperature(&logits, 0.5);
    assert!(sharp[2] > logits[2]); // Larger logits amplified more

    // Temperature > 1 should reduce differences
    let soft = apply_temperature(&logits, 2.0);
    assert!(soft[2] < logits[2]); // Larger logits reduced more
}

#[test]
fn test_softmax_properties() {
    let logits = vec![1.0, 2.0, 3.0];
    let probs = softmax(&logits);

    // All probabilities should be positive
    for &p in &probs {
        assert!(p > 0.0);
        assert!(p <= 1.0);
    }

    // Should sum to approximately 1
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);

    // Higher logits should have higher probability
    assert!(probs[2] > probs[1]);
    assert!(probs[1] > probs[0]);
}

#[test]
fn test_softmax_numerical_stability() {
    // Very large logits that could cause overflow
    let large_logits = vec![1000.0, 1001.0, 1002.0];
    let probs = softmax(&large_logits);

    // Should still sum to 1
    let sum: f32 = probs.iter().sum();
    assert!((sum - 1.0).abs() < 0.001);

    // All should be finite
    for &p in &probs {
        assert!(p.is_finite());
    }
}

#[test]
fn test_sample_greedy() {
    let logits = vec![0.1, 0.5, 0.3, 0.8];
    let token = sample_greedy(&logits).unwrap();
    assert_eq!(token, 3); // Index of max value
}

#[test]
fn test_sample_greedy_empty() {
    let logits = vec![];
    let result = sample_greedy(&logits);
    assert!(result.is_err());
}

#[test]
fn test_sample_temperature() {
    let logits = vec![1.0, 2.0, 3.0];
    let config = TemperatureConfig::standard();
    let token = sample_temperature(&logits, &config, 42).unwrap();
    assert!(token < logits.len());
}

#[test]
fn test_sample_top_k_standard() {
    let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
    let config = TopKConfig::standard();
    let token = sample_top_k(&logits, &config, 42).unwrap();
    assert!(token < logits.len());
}

#[test]
fn test_sample_top_k_invalid_k() {
    let logits = vec![0.1, 0.5, 0.3];
    let config = TopKConfig {
        k: 0,
        temperature: 1.0,
    };
    let result = sample_top_k(&logits, &config, 42);
    assert!(result.is_err());
}

#[test]
fn test_sample_top_p_standard() {
    let logits = vec![0.1, 0.5, 0.3, 0.8, 0.2];
    let config = TopPConfig::standard();
    let token = sample_top_p(&logits, &config, 42).unwrap();
    assert!(token < logits.len());
}

#[test]
fn test_sample_top_p_invalid_p() {
    let logits = vec![0.1, 0.5, 0.3];
    let config = TopPConfig {
        p: 1.5,
        temperature: 1.0,
    };
    let result = sample_top_p(&logits, &config, 42);
    assert!(result.is_err());
}

#[test]
fn test_top_k_config_standard() {
    let config = TopKConfig::standard();
    assert_eq!(config.k, 50);
    assert_eq!(config.temperature, 1.0);
}

#[test]
fn test_top_p_config_standard() {
    let config = TopPConfig::standard();
    assert!((config.p - 0.95).abs() < 0.001);
    assert_eq!(config.temperature, 1.0);
}
