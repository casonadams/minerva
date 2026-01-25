use minerva::inference::engine_config::InferenceEngineConfig;

#[test]
fn test_llama_config_creation() {
    let config = InferenceEngineConfig::llama(32000, 4096, 32);
    assert_eq!(config.vocab_size, 32000);
    assert_eq!(config.hidden_size, 4096);
    assert_eq!(config.num_heads, 32);
    assert_eq!(config.num_layers, 32);
    assert!(config.causal);
}

#[test]
fn test_bert_config_creation() {
    let config = InferenceEngineConfig::bert(30522, 768, 12);
    assert_eq!(config.vocab_size, 30522);
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_heads, 12);
    assert!(!config.causal); // BERT is non-causal
}

#[test]
fn test_tiny_config_creation() {
    let config = InferenceEngineConfig::tiny(1000);
    assert_eq!(config.vocab_size, 1000);
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.hidden_size, 64);
    assert_eq!(config.num_heads, 2);
}

#[test]
fn test_llama_intermediate_size() {
    let config = InferenceEngineConfig::llama(32000, 4096, 32);
    // LLaMA uses 8/3 ratio
    assert_eq!(config.intermediate_size, (4096 * 8) / 3);
}

#[test]
fn test_bert_intermediate_size() {
    let config = InferenceEngineConfig::bert(30522, 768, 12);
    // BERT uses 4x ratio
    assert_eq!(config.intermediate_size, 768 * 4);
}
