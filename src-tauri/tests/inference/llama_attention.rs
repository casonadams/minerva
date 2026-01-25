use minerva::inference::llama_attention::{AttentionOutput, MultiHeadAttention};

#[test]
fn test_multihead_attention_creation() {
    let attn = MultiHeadAttention::new(8, 512).unwrap();
    assert_eq!(attn.num_heads, 8);
    assert_eq!(attn.head_dim, 64);
}

#[test]
fn test_multihead_attention_invalid_dims() {
    let result = MultiHeadAttention::new(8, 510);
    assert!(result.is_err());
}

#[test]
fn test_attention_output() {
    let output = AttentionOutput {
        output: vec![0.1, 0.2, 0.3],
        weights: Some(vec![0.5, 0.5]),
    };
    assert_eq!(output.output.len(), 3);
    assert!(output.weights.is_some());
}

#[test]
fn test_multihead_attention_forward() {
    let attn = MultiHeadAttention::new(2, 64).unwrap();
    let mut query = vec![0.1; 64];
    let mut key = vec![0.2; 64];
    let value = vec![0.3; 64];

    let params = minerva::inference::llama_attention::AttentionParams {
        query: &mut query,
        key: &mut key,
        value: &value,
        pos: 0,
    };

    let result = attn.forward(params);
    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.output.len(), 64);
    assert!(output.weights.is_some());
}
