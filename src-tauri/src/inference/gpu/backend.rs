use crate::error::MinervaResult;
use crate::inference::gpu::layers;
use crate::inference::gpu::loader::{AttentionWeights, MLPWeights};
use crate::inference::gpu::{KVCache, ModelConfig, SafeTensorsLoader};
use ndarray::Array2;
/// GPU SafeTensors Backend
///
/// Main inference engine that loads SafeTensors models and performs forward passes.
/// Day 1 implementation: Basic GPU backend with KV cache placeholder
use std::path::Path;

pub struct GPUSafeTensorsBackend {
    pub config: ModelConfig,

    // Weights - stored as ndarray for efficiency
    pub embedding: Array2<f32>,
    pub norm_weight: Array2<f32>,
    pub lm_head: Array2<f32>,

    // Per-layer weights (simplified - store first few layers for now)
    pub attention_layers: Vec<AttentionWeights>,
    pub mlp_layers: Vec<MLPWeights>,
    pub attn_norm_weights: Vec<Array2<f32>>,
    pub ffn_norm_weights: Vec<Array2<f32>>,

    pub model_path: std::path::PathBuf,
}

impl GPUSafeTensorsBackend {
    /// Create new GPU backend
    pub fn new(model_path: &Path, config_path: &Path) -> MinervaResult<Self> {
        println!("Loading model configuration...");
        let config = ModelConfig::from_file(config_path)?;

        println!("Loading weights from SafeTensors...");

        // Load embedding
        println!("  Loading embedding...");
        let embedding = SafeTensorsLoader::load_embedding(model_path)?;
        println!("    Embedding shape: {:?}", embedding.shape());

        // Load final weights
        println!("  Loading final layer norm and LM head...");
        let (norm_weight, lm_head) = SafeTensorsLoader::load_final_weights(model_path)?;
        println!(
            "    Norm shape: {:?}, LM head shape: {:?}",
            norm_weight.shape(),
            lm_head.shape()
        );

        // Load layer-specific weights (all layers for now)
        let mut attention_layers = Vec::new();
        let mut mlp_layers = Vec::new();
        let mut attn_norm_weights = Vec::new();
        let mut ffn_norm_weights = Vec::new();

        println!("  Loading {} layers...", config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            if (layer_idx + 1) % 8 == 0 {
                println!("    Layer {}/{}", layer_idx + 1, config.num_hidden_layers);
            }

            let attn = SafeTensorsLoader::load_attention_projections(model_path, layer_idx)?;
            if layer_idx == 0 {
                println!(
                    "    Layer 0 attention shapes - Q: {:?}, K: {:?}, V: {:?}, O: {:?}",
                    attn.q_proj.shape(),
                    attn.k_proj.shape(),
                    attn.v_proj.shape(),
                    attn.o_proj.shape()
                );
            }
            attention_layers.push(attn);

            let mlp = SafeTensorsLoader::load_mlp_weights(model_path, layer_idx)?;
            mlp_layers.push(mlp);

            let (attn_norm, ffn_norm) =
                SafeTensorsLoader::load_norm_weights(model_path, layer_idx)?;
            attn_norm_weights.push(attn_norm);
            ffn_norm_weights.push(ffn_norm);
        }

        println!("✅ Model loaded successfully!");
        println!("   Architecture: {}", config.architectures.join(", "));
        println!("   Hidden size: {}", config.hidden_size);
        println!("   Num layers: {}", config.num_hidden_layers);
        println!("   Vocab size: {}", config.vocab_size);

        Ok(Self {
            config,
            embedding,
            norm_weight,
            lm_head,
            attention_layers,
            mlp_layers,
            attn_norm_weights,
            ffn_norm_weights,
            model_path: model_path.to_path_buf(),
        })
    }

    /// Forward pass through model
    pub fn forward(
        &self,
        token_ids: &[usize],
        _kv_cache: &mut Option<KVCache>,
    ) -> MinervaResult<Array2<f32>> {
        // Convert token IDs to embeddings
        let mut hidden = self.embedding_lookup(token_ids)?;

        // Pass through all transformer layers
        for layer_idx in 0..self.config.num_hidden_layers {
            hidden = layers::transformer_layer(
                &hidden,
                &self.attention_layers[layer_idx].q_proj,
                &self.attention_layers[layer_idx].k_proj,
                &self.attention_layers[layer_idx].v_proj,
                &self.attention_layers[layer_idx].o_proj,
                &self.mlp_layers[layer_idx].gate_proj,
                &self.mlp_layers[layer_idx].up_proj,
                &self.mlp_layers[layer_idx].down_proj,
                &self.attn_norm_weights[layer_idx],
                &self.ffn_norm_weights[layer_idx],
                self.config.num_attention_heads,
                self.config.rms_norm_eps,
                _kv_cache,
            );
        }

        // Final normalization
        let hidden = layers::rms_norm(&hidden, &self.norm_weight, self.config.rms_norm_eps);

        // Project to vocab
        let logits = hidden.dot(&self.lm_head);

        Ok(logits)
    }

    /// Embedding lookup from token IDs
    fn embedding_lookup(&self, token_ids: &[usize]) -> MinervaResult<Array2<f32>> {
        // Stack embeddings: (seq_len, hidden_size)
        let mut result = ndarray::Array2::zeros((token_ids.len(), self.config.hidden_size));

        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id >= self.embedding.shape()[0] {
                eprintln!(
                    "Token ID {} out of range (vocab: {})",
                    token_id,
                    self.embedding.shape()[0]
                );
                // Keep as zeros
            } else {
                let row = self.embedding.row(token_id);
                for j in 0..self.config.hidden_size.min(row.len()) {
                    result[[i, j]] = row[j];
                }
            }
        }

        Ok(result)
    }

    /// Get model configuration
    pub fn get_config(&self) -> &ModelConfig {
        &self.config
    }

    /// Get model statistics
    pub fn get_stats(&self) -> BackendStats {
        let total_params = (self.embedding.shape()[0] * self.embedding.shape()[1])
            + (self.lm_head.shape()[0] * self.lm_head.shape()[1])
            + self
                .attention_layers
                .iter()
                .map(|a| {
                    a.q_proj.shape()[0] * a.q_proj.shape()[1]
                        + a.k_proj.shape()[0] * a.k_proj.shape()[1]
                        + a.v_proj.shape()[0] * a.v_proj.shape()[1]
                        + a.o_proj.shape()[0] * a.o_proj.shape()[1]
                })
                .sum::<usize>()
            + self
                .mlp_layers
                .iter()
                .map(|m| {
                    m.gate_proj.shape()[0] * m.gate_proj.shape()[1]
                        + m.up_proj.shape()[0] * m.up_proj.shape()[1]
                        + m.down_proj.shape()[0] * m.down_proj.shape()[1]
                })
                .sum::<usize>();

        BackendStats {
            total_parameters: total_params,
            num_layers: self.config.num_hidden_layers,
            hidden_size: self.config.hidden_size,
            vocab_size: self.config.vocab_size,
        }
    }
}

pub struct BackendStats {
    pub total_parameters: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Only run when model is available
    fn test_backend_creation() {
        use std::time::Instant;

        let model_path = Path::new("../models/tinyllama-1.1b-safetensors/model.safetensors");
        let config_path = Path::new("../models/tinyllama-1.1b-safetensors/config.json");

        println!("\n=== Testing GPU Backend ===\n");

        // Test 1: Load model
        println!("Step 1: Loading model...");
        let start = Instant::now();
        let result = GPUSafeTensorsBackend::new(model_path, config_path);
        let load_time = start.elapsed();

        assert!(result.is_ok(), "Failed to load model");
        println!("✓ Model loaded in {:.2}s", load_time.as_secs_f32());

        let backend = result.unwrap();
        let stats = backend.get_stats();
        assert!(stats.total_parameters > 0);

        println!("\nModel Statistics:");
        println!(
            "  Total parameters: {} ({:.1}B)",
            stats.total_parameters,
            stats.total_parameters as f64 / 1e9
        );
        println!("  Num layers: {}", stats.num_layers);
        println!("  Hidden size: {}", stats.hidden_size);
        println!("  Vocab size: {}", stats.vocab_size);

        // Note: Full forward pass requires implementing GQA (Grouped Query Attention)
        // TinyLlama uses GQA: num_heads=32, num_kv_heads=8
        // This requires separate KV cache handling and attention computation
        // Will be implemented in Day 2 when adding Flash Attention

        println!("\nStep 2: Forward pass test");
        println!("Status: SKIPPED (requires GQA support)");
        println!("  Model uses Grouped Query Attention (GQA)");
        println!("  - num_attention_heads: 32");
        println!("  - num_key_value_heads: 8");
        println!("  - This requires special handling in attention computation");
        println!("  - Will implement in Day 2 with Flash Attention");

        println!("\n=== GPU Backend Test Passed (Model Loading) ===");
        println!("  Status: Ready for Day 2 - Forward Pass Implementation");
        println!("  Next: Implement GQA attention and KV cache handling\n");
    }
}
