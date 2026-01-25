use minerva_lib::inference::embedding::{embed_tokens, EmbeddingConfig};

#[test]
fn test_embed_tokens_shapes() {
    let vocab_size = 10;
    let hidden_size = 4;
    let embeddings = vec![0.1; vocab_size * hidden_size];
    let tokens = vec![0, 1, 2];

    let config = EmbeddingConfig {
        vocab_size,
        hidden_size,
    };
    let output = embed_tokens(&tokens, &embeddings, &config).unwrap();

    assert_eq!(output.len(), tokens.len() * hidden_size);
}

#[test]
fn test_embed_tokens_out_of_vocab() {
    let vocab_size = 10;
    let hidden_size = 4;
    let embeddings = vec![0.1; vocab_size * hidden_size];
    let tokens = vec![0, 999]; // 999 is out of vocab

    let config = EmbeddingConfig {
        vocab_size,
        hidden_size,
    };
    let result = embed_tokens(&tokens, &embeddings, &config);
    assert!(result.is_err());
}

#[test]
fn test_embed_tokens_values() {
    let vocab_size = 5;
    let hidden_size = 2;
    let embeddings = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    let tokens = vec![0, 2];

    let config = EmbeddingConfig {
        vocab_size,
        hidden_size,
    };
    let output = embed_tokens(&tokens, &embeddings, &config).unwrap();

    // First token: [0.1, 0.2]
    assert!((output[0] - 0.1).abs() < 0.01);
    assert!((output[1] - 0.2).abs() < 0.01);
    // Second token: [0.5, 0.6]
    assert!((output[2] - 0.5).abs() < 0.01);
    assert!((output[3] - 0.6).abs() < 0.01);
}
