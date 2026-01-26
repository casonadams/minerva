#[cfg(test)]
mod tests {
    use crate::inference::mlx_native::config::GPTOSSConfig;
    use crate::inference::mlx_native::loader::*;

    #[test]
    fn test_config_extraction() {
        let config = GPTOSSConfig::default();
        assert_eq!(config.vocab_size, 201088);
        assert_eq!(config.hidden_size, 2880);
        assert_eq!(config.num_hidden_layers, 24);
    }

    #[test]
    #[ignore] // Requires model files
    fn test_load_mlx_gpt_oss_20b() {
        let model_path = std::path::PathBuf::from(
            std::path::PathBuf::from(env!("HOME")).join(".cache/mlx-models/gpt-oss-20b-MXFP4-Q8"),
        );

        if !model_path.exists() {
            eprintln!(
                "Model not found at {:?}, skipping test. Download from HuggingFace.",
                model_path
            );
            return;
        }

        let model = load_mlx_model(&model_path).expect("Failed to load model");

        // Verify structure
        assert_eq!(model.embedding.shape()[0], 201088);
        assert_eq!(model.embedding.shape()[1], 2880);
        assert_eq!(model.layers.len(), 24);
        assert_eq!(model.lm_head.shape()[0], 201088);

        println!("✓ Loaded model with {} layers", model.num_layers());
        println!("✓ Embedding shape: {:?}", model.embedding.shape());
        println!("✓ Vocab size: {}", model.vocab_size());
    }
}
