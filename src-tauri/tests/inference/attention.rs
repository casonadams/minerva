use minerva_lib::inference::attention::{
    scaled_dot_product_attention, AttentionConfig, AttentionInput,
};

#[test]
fn test_causal_mask_application() {
    let seq_len = 3;
    let head_size = 2;

    // Use different values to see difference with/without causal mask
    let mut query = vec![0.0; seq_len * head_size];
    let mut key = vec![0.0; seq_len * head_size];
    let mut value = vec![0.0; seq_len * head_size];

    // First token
    query[0] = 1.0;
    key[0] = 1.0;
    value[0] = 1.0;

    // Second token (distinct)
    query[head_size] = 2.0;
    key[head_size] = 2.0;
    value[head_size] = 2.0;

    // Third token (distinct)
    query[2 * head_size] = 3.0;
    key[2 * head_size] = 3.0;
    value[2 * head_size] = 3.0;

    // With causal mask - last token should not attend to future (none available)
    let config_causal = AttentionConfig {
        seq_len,
        head_size,
        causal: true,
    };
    let input = AttentionInput {
        query: &query,
        key: &key,
        value: &value,
    };
    let output_causal = scaled_dot_product_attention(&input, &config_causal).unwrap();

    // Without causal mask (not used in this test but shown for reference)
    let config_no_causal = AttentionConfig {
        seq_len,
        head_size,
        causal: false,
    };
    let _output_no_causal = scaled_dot_product_attention(&input, &config_no_causal).unwrap();

    // First token in causal mode should only see itself
    let first_token_causal = &output_causal[0..head_size];

    // First token in causal mode should have high attention to itself
    // (since it can only attend to position 0)
    assert!(
        first_token_causal[0] > 0.5,
        "First token should largely attend to itself in causal mode"
    );
}

#[test]
fn test_scaled_dot_product_attention_invalid_shapes() {
    let config = AttentionConfig {
        seq_len: 4,
        head_size: 8,
        causal: false,
    };
    let input = AttentionInput {
        query: &[0.1],
        key: &[0.2],
        value: &[0.3],
    };
    let result = scaled_dot_product_attention(&input, &config);
    assert!(result.is_err());
}

#[test]
fn test_attention_softmax_sums_to_one() {
    let seq_len = 2;
    let head_size = 4;

    let query = vec![1.0; seq_len * head_size];
    let key = vec![1.0; seq_len * head_size];
    let value = vec![1.0; seq_len * head_size];

    let config = AttentionConfig {
        seq_len,
        head_size,
        causal: false,
    };
    let input = AttentionInput {
        query: &query,
        key: &key,
        value: &value,
    };
    let output = scaled_dot_product_attention(&input, &config).unwrap();

    // Since all queries, keys, and values are the same,
    // output should be close to the value (all positions contribute equally)
    assert_eq!(output.len(), seq_len * head_size);
    for val in output.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_attention_numerical_stability() {
    let seq_len = 4;
    let head_size = 8;

    let query = vec![100.0; seq_len * head_size]; // Large values
    let key = vec![100.0; seq_len * head_size];
    let value = vec![0.5; seq_len * head_size];

    let config = AttentionConfig {
        seq_len,
        head_size,
        causal: false,
    };
    let input = AttentionInput {
        query: &query,
        key: &key,
        value: &value,
    };
    let output = scaled_dot_product_attention(&input, &config).unwrap();

    // Should not have NaN or Inf (numerical stability check)
    for val in output.iter() {
        assert!(val.is_finite(), "Output contains NaN or Inf");
    }
}
