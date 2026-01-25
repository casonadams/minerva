use minerva::inference::llama_feedforward::{FFParams, FeedForward};

#[test]
fn test_feedforward_creation() {
    let ff = FeedForward::new(512, 2048);
    assert_eq!(ff.hidden_size, 512);
    assert_eq!(ff.intermediate_size, 2048);
}

#[test]
fn test_feedforward_forward() {
    let ff = FeedForward::new(4, 8);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let up_w = vec![0.1; 32];
    let down_w = vec![0.1; 32];

    let result = ff
        .forward(FFParams {
            x: &x,
            up_weight: &up_w,
            down_weight: &down_w,
        })
        .unwrap();
    assert_eq!(result.len(), 4);
}

#[test]
fn test_feedforward_size_mismatch() {
    let ff = FeedForward::new(4, 8);
    let x = vec![1.0, 2.0]; // Wrong size
    let up_w = vec![0.1; 32];
    let down_w = vec![0.1; 32];

    let result = ff.forward(FFParams {
        x: &x,
        up_weight: &up_w,
        down_weight: &down_w,
    });
    assert!(result.is_err());
}

#[test]
fn test_feedforward_weight_mismatch() {
    let ff = FeedForward::new(4, 8);
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let up_w = vec![0.1; 16]; // Wrong size
    let down_w = vec![0.1; 32];

    let result = ff.forward(FFParams {
        x: &x,
        up_weight: &up_w,
        down_weight: &down_w,
    });
    assert!(result.is_err());
}
