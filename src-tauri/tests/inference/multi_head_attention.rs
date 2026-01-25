use minerva_lib::inference::multi_head_attention::{multi_head_attention, MultiHeadConfig};

#[test]
fn test_multi_head_attention_shapes() {
    let seq_len = 4;
    let hidden_size = 16;
    let num_heads = 4;

    let input = vec![0.1; seq_len * hidden_size];
    let config = MultiHeadConfig {
        seq_len,
        hidden_size,
        num_heads,
        causal: false,
    };
    let output = multi_head_attention(&input, &config).unwrap();

    assert_eq!(output.len(), seq_len * hidden_size);
}

#[test]
fn test_multi_head_attention_invalid_heads() {
    let seq_len = 4;
    let hidden_size = 15; // Not divisible by num_heads
    let num_heads = 4;

    let input = vec![0.1; seq_len * hidden_size];
    let config = MultiHeadConfig {
        seq_len,
        hidden_size,
        num_heads,
        causal: false,
    };
    let result = multi_head_attention(&input, &config);

    assert!(result.is_err());
}

#[test]
fn test_multi_head_attention_single_head() {
    let seq_len = 3;
    let hidden_size = 8;
    let num_heads = 1;

    let input = vec![0.5; seq_len * hidden_size];
    let config = MultiHeadConfig {
        seq_len,
        hidden_size,
        num_heads,
        causal: false,
    };
    let output = multi_head_attention(&input, &config).unwrap();

    assert_eq!(output.len(), seq_len * hidden_size);
    for val in output.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_multi_head_attention_many_heads() {
    let seq_len = 2;
    let hidden_size = 32;
    let num_heads = 8;

    let input = vec![1.0; seq_len * hidden_size];
    let config = MultiHeadConfig {
        seq_len,
        hidden_size,
        num_heads,
        causal: false,
    };
    let output = multi_head_attention(&input, &config).unwrap();

    assert_eq!(output.len(), seq_len * hidden_size);
    for val in output.iter() {
        assert!(val.is_finite());
    }
}

#[test]
fn test_multi_head_attention_causal() {
    let seq_len = 4;
    let hidden_size = 8;
    let num_heads = 2;

    let input = vec![0.1; seq_len * hidden_size];
    let config = MultiHeadConfig {
        seq_len,
        hidden_size,
        num_heads,
        causal: true,
    };
    let output = multi_head_attention(&input, &config).unwrap();

    assert_eq!(output.len(), seq_len * hidden_size);
    for val in output.iter() {
        assert!(val.is_finite());
    }
}
