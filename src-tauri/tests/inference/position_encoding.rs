use minerva_lib::inference::position_encoding::{
    add_position_encoding, create_position_encoding, PositionConfig,
};

#[test]
fn test_position_encoding_shapes() {
    let config = PositionConfig {
        seq_len: 4,
        hidden_size: 8,
        base: 10000.0,
    };
    let pe = create_position_encoding(&config);
    assert_eq!(pe.len(), 4 * 8);
}

#[test]
fn test_position_encoding_symmetry() {
    let config = PositionConfig {
        seq_len: 3,
        hidden_size: 4,
        base: 10000.0,
    };
    let pe = create_position_encoding(&config);

    // Check that even/odd pairs follow sin/cos pattern
    // Position 0, dimension 0 should be sin
    let pos0_dim0 = pe[0];
    let pos0_dim1 = pe[1];

    // Both should be close to their expected values
    // sin(0) = 0, cos(0) = 1
    assert!(pos0_dim0.abs() < 0.01);
    assert!((pos0_dim1 - 1.0).abs() < 0.01);
}

#[test]
fn test_add_position_encoding() {
    let embeddings = vec![0.5; 12]; // 3 tokens × 4 hidden
    let pe = vec![0.1; 12];

    let result = add_position_encoding(&embeddings, &pe).unwrap();
    assert_eq!(result.len(), 12);
    assert!((result[0] - 0.6).abs() < 0.01); // 0.5 + 0.1
}

#[test]
fn test_add_position_encoding_shape_mismatch() {
    let embeddings = vec![0.5; 12];
    let pe = vec![0.1; 8]; // Wrong shape

    let result = add_position_encoding(&embeddings, &pe);
    assert!(result.is_err());
}
