use minerva::inference::llama_utils::{rmsnorm, silu};

#[test]
fn test_rmsnorm_basic() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let weight = vec![0.5, 0.5, 0.5, 0.5];
    let result = rmsnorm(&x, &weight, 1e-6).unwrap();
    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|v| v.is_finite()));
}

#[test]
fn test_rmsnorm_size_mismatch() {
    let x = vec![1.0, 2.0];
    let weight = vec![0.5, 0.5, 0.5];
    let result = rmsnorm(&x, &weight, 1e-6);
    assert!(result.is_err());
}

#[test]
fn test_rmsnorm_uniform_input() {
    let x = vec![2.0; 10];
    let weight = vec![1.0; 10];
    let result = rmsnorm(&x, &weight, 1e-6).unwrap();
    assert!(result.iter().all(|v| (v - 1.0).abs() < 1e-4));
}

#[test]
fn test_silu_basic() {
    let x = vec![0.0, 1.0, -1.0, 2.0];
    let result = silu(&x);
    assert_eq!(result.len(), 4);
    assert!(result.iter().all(|v| v.is_finite()));
}

#[test]
fn test_silu_zero() {
    let x = vec![0.0];
    let result = silu(&x);
    assert!((result[0] - 0.0).abs() < 1e-5);
}

#[test]
fn test_silu_positive() {
    let x = vec![1.0];
    let result = silu(&x);
    assert!(result[0] > 0.7);
    assert!(result[0] < 1.0);
}

#[test]
fn test_silu_range() {
    let x = vec![0.0, 1.0, -1.0, 2.0];
    let result = silu(&x);
    assert!(result[0] >= 0.0 && result[0] <= 0.1);
    assert!(result[1] > 0.7);
}
