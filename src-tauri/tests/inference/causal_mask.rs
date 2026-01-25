use minerva_lib::inference::causal_mask::create_causal_mask;

#[test]
fn test_causal_mask_creation() {
    let seq_len = 3;
    let mask = create_causal_mask(seq_len);

    assert_eq!(mask.len(), seq_len * seq_len);

    // Position 0 can attend to position 0 only
    assert!(mask[0 * seq_len + 0]);
    assert!(!mask[0 * seq_len + 1]);
    assert!(!mask[0 * seq_len + 2]);

    // Position 1 can attend to positions 0, 1
    assert!(mask[1 * seq_len + 0]);
    assert!(mask[1 * seq_len + 1]);
    assert!(!mask[1 * seq_len + 2]);

    // Position 2 can attend to positions 0, 1, 2
    assert!(mask[2 * seq_len + 0]);
    assert!(mask[2 * seq_len + 1]);
    assert!(mask[2 * seq_len + 2]);
}

#[test]
fn test_causal_mask_single_position() {
    let seq_len = 1;
    let mask = create_causal_mask(seq_len);

    assert_eq!(mask.len(), 1);
    assert!(mask[0]);
}

#[test]
fn test_causal_mask_large_sequence() {
    let seq_len = 10;
    let mask = create_causal_mask(seq_len);

    assert_eq!(mask.len(), seq_len * seq_len);

    // Verify pattern for several positions
    for i in 0..seq_len {
        for j in 0..seq_len {
            let expected = j <= i;
            let actual = mask[i * seq_len + j];
            assert_eq!(
                actual, expected,
                "Position ({}, {}) has wrong mask value",
                i, j
            );
        }
    }
}
