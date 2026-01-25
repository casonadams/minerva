use minerva::inference::inference_backend_trait::GenerationParams;

#[test]
fn test_generation_params_creation() {
    let params = GenerationParams {
        max_tokens: 100,
        temperature: 0.7,
        top_p: 0.9,
    };
    assert_eq!(params.max_tokens, 100);
    assert!((params.temperature - 0.7).abs() < 0.01);
    assert!((params.top_p - 0.9).abs() < 0.01);
}

#[test]
fn test_generation_params_default() {
    let params = GenerationParams {
        max_tokens: 256,
        temperature: 1.0,
        top_p: 0.95,
    };
    assert_eq!(params.max_tokens, 256);
}
