use minerva::inference::rope_utils::RoPEParams;

#[test]
fn test_rope_params_creation() {
    let rope = RoPEParams::new(64);
    assert_eq!(rope.head_dim, 64);
}

#[test]
fn test_rope_angle_zero_position() {
    let rope = RoPEParams::new(64);
    let angle = rope.get_angle(0, 0);
    assert_eq!(angle, 0.0);
}

#[test]
fn test_rope_angle_positive_position() {
    let rope = RoPEParams::new(64);
    let angle = rope.get_angle(1, 0);
    assert!(angle > 0.0);
}

#[test]
fn test_rope_different_head_dims() {
    let rope_32 = RoPEParams::new(32);
    let rope_128 = RoPEParams::new(128);
    assert_eq!(rope_32.head_dim, 32);
    assert_eq!(rope_128.head_dim, 128);
}
