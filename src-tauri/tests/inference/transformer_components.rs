use minerva_lib::inference::transformer_components::{
    transformer_block, Activation, TransformerBlockConfig, TransformerBlockWeights,
};

#[test]
fn test_transformer_block_shapes() {
    let seq_len = 2;
    let hidden_size = 8;
    let num_heads = 2;
    let intermediate_size = 16;

    let input = vec![0.1; seq_len * hidden_size];
    let attn_scale = vec![1.0; hidden_size];
    let ff_up = vec![0.2; hidden_size * intermediate_size];
    let ff_down = vec![0.1; intermediate_size * hidden_size];

    let config = TransformerBlockConfig {
        seq_len,
        hidden_size,
        num_heads,
        intermediate_size,
        activation: Activation::GELU,
        causal: true,
        eps: 1e-6,
    };

    let weights = TransformerBlockWeights {
        attn_scale: Some(&attn_scale),
        ff_up: &ff_up,
        ff_down: &ff_down,
    };

    let output = transformer_block(&input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * hidden_size);
}

#[test]
fn test_transformer_block_residuals() {
    let seq_len = 1;
    let hidden_size = 4;
    let num_heads = 1;
    let intermediate_size = 8;

    let input = vec![1.0; seq_len * hidden_size];
    let attn_scale = vec![1.0; hidden_size];
    let ff_up = vec![0.1; hidden_size * intermediate_size];
    let ff_down = vec![0.1; intermediate_size * hidden_size];

    let config = TransformerBlockConfig {
        seq_len,
        hidden_size,
        num_heads,
        intermediate_size,
        activation: Activation::ReLU,
        causal: false,
        eps: 1e-6,
    };

    let weights = TransformerBlockWeights {
        attn_scale: Some(&attn_scale),
        ff_up: &ff_up,
        ff_down: &ff_down,
    };

    let output = transformer_block(&input, &weights, &config).unwrap();
    assert_eq!(output.len(), input.len());
    for val in output.iter() {
        assert!(val.is_finite());
    }
}
