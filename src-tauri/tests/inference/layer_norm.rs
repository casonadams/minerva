use minerva_lib::inference::layer_norm::{layer_norm, LayerNormConfig};

#[test]
fn test_layer_norm_normalization() {
    let seq_len = 2;
    let hidden_size = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let config = LayerNormConfig {
        seq_len,
        hidden_size,
        scale: None,
        eps: 1e-6,
    };
    let output = layer_norm(&input, &config).unwrap();
    assert_eq!(output.len(), seq_len * hidden_size);
    // Output should be normalized
    for chunk in output.chunks(hidden_size) {
        let mean_sq: f32 = chunk.iter().map(|x| x * x).sum::<f32>() / hidden_size as f32;
        let rms = mean_sq.sqrt();
        assert!((rms - 1.0).abs() < 0.1, "RMS should be close to 1.0");
    }
}

#[test]
fn test_layer_norm_shapes() {
    let seq_len = 3;
    let hidden_size = 8;
    let input = vec![0.5; seq_len * hidden_size];
    let config = LayerNormConfig {
        seq_len,
        hidden_size,
        scale: None,
        eps: 1e-6,
    };
    let output = layer_norm(&input, &config).unwrap();
    assert_eq!(output.len(), seq_len * hidden_size);
}

#[test]
fn test_layer_norm_with_scale() {
    let seq_len = 1;
    let hidden_size = 4;
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let scale = vec![2.0, 2.0, 2.0, 2.0];
    let config = LayerNormConfig {
        seq_len,
        hidden_size,
        scale: Some(scale),
        eps: 1e-6,
    };
    let output = layer_norm(&input, &config).unwrap();
    assert_eq!(output.len(), hidden_size);
}
