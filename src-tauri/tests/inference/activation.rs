use minerva_lib::inference::activation::{apply_activation, gelu, relu, silu, Activation};

#[test]
fn test_gelu() {
    let x = 1.0;
    let result = gelu(x);
    assert!(result > 0.0 && result < x);
}

#[test]
fn test_silu() {
    let x = 1.0;
    let result = silu(x);
    assert!(result > 0.0 && result < 1.0);
}

#[test]
fn test_relu() {
    assert_eq!(relu(1.0), 1.0);
    assert_eq!(relu(-1.0), 0.0);
}

#[test]
fn test_apply_activation_gelu() {
    let input = vec![1.0, -1.0, 0.5];
    let result = apply_activation(&input, Activation::GELU);
    assert_eq!(result.len(), 3);
    assert!(result[0] > 0.0);
}

#[test]
fn test_apply_activation_silu() {
    let input = vec![1.0, -1.0, 0.5];
    let result = apply_activation(&input, Activation::SiLU);
    assert_eq!(result.len(), 3);
}

#[test]
fn test_apply_activation_relu() {
    let input = vec![1.0, -1.0, 0.5];
    let result = apply_activation(&input, Activation::ReLU);
    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 0.0);
    assert_eq!(result[2], 0.5);
}
