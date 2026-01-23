use super::gpu_compute_engine::{AttentionParams, GPUComputeEngine, MatmulParams, RmsNormParams};
/// GPU-Integrated LLaMA Inference - Phase 6 Step 4
///
/// This module integrates the GPU Computation Engine with LLaMA inference,
/// providing GPU-accelerated operations for:
/// - Attention computation with GPU kernels
/// - Feed-forward network computation with GPU
/// - Layer normalization with GPU support
/// - Full inference pipeline with GPU/CPU fallback
use crate::error::{MinervaError, MinervaResult};

/// GPU Inference Configuration
#[derive(Debug, Clone)]
pub struct GPUInferenceConfig {
    /// Enable GPU acceleration
    pub gpu_enabled: bool,
    /// Use simulated GPU (for testing)
    pub use_simulation: bool,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Model hidden dimension
    pub hidden_dim: usize,
    /// Intermediate dimension for FFN
    pub intermediate_dim: usize,
}

impl Default for GPUInferenceConfig {
    fn default() -> Self {
        Self {
            gpu_enabled: true,
            use_simulation: true,
            num_heads: 8,
            head_dim: 64,
            hidden_dim: 512,
            intermediate_dim: 2048,
        }
    }
}

/// GPU Inference Result
#[derive(Debug, Clone)]
pub struct GPUInferenceResult {
    /// Output embeddings
    pub output: Vec<f32>,
    /// Attention result
    pub attention_time_ms: f32,
    /// FFN result
    pub ffn_time_ms: f32,
    /// Total computation time
    pub total_time_ms: f32,
    /// GPU used flag
    pub used_gpu: bool,
}

/// GPU-Integrated LLaMA Inference Engine
pub struct GPULlamaInference {
    /// GPU compute engine
    compute_engine: GPUComputeEngine,
    /// Configuration
    config: GPUInferenceConfig,
}

impl GPULlamaInference {
    /// Create new GPU-integrated LLaMA inference
    pub fn new(config: GPUInferenceConfig) -> MinervaResult<Self> {
        let compute_engine = GPUComputeEngine::simulated()?;
        Ok(Self {
            compute_engine,
            config,
        })
    }

    /// Create simulated GPU-integrated inference (for testing)
    pub fn simulated() -> MinervaResult<Self> {
        Self::new(GPUInferenceConfig::default())
    }

    /// Execute single transformer block with GPU acceleration
    pub fn forward_block(
        &self,
        input: &[f32],
        q_weight: &[f32],
        k_weight: &[f32],
        v_weight: &[f32],
        o_weight: &[f32],
        ffn_up: &[f32],
        ffn_down: &[f32],
        ffn_gate: &[f32],
        norm_weight: &[f32],
    ) -> MinervaResult<GPUInferenceResult> {
        let total_start = std::time::Instant::now();

        // Validate input shape
        if input.len() != self.config.hidden_dim {
            return Err(MinervaError::InferenceError(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.config.hidden_dim,
                input.len()
            )));
        }

        // Layer normalization
        let norm_params = RmsNormParams::new(input.to_vec(), norm_weight.to_vec(), 1e-6);
        let norm_result = self.compute_engine.compute_rmsnorm(norm_params)?;

        // Attention computation
        let attention_start = std::time::Instant::now();
        let q = self.project_to_attention(&norm_result.output, q_weight)?;
        let k = self.project_to_attention(&norm_result.output, k_weight)?;
        let v = self.project_to_attention(&norm_result.output, v_weight)?;

        let attn_params = AttentionParams::new(q, k, v, self.config.num_heads);
        let attn_result = self.compute_engine.compute_attention(attn_params)?;
        let attention_time = attention_start.elapsed().as_secs_f32() * 1000.0;

        // Output projection
        let attn_out = self.project_output(&attn_result.output, o_weight)?;

        // Residual connection
        let residual = self.add_residual(input, &attn_out);

        // FFN computation
        let ffn_start = std::time::Instant::now();
        let norm_params2 = RmsNormParams::new(residual.clone(), norm_weight.to_vec(), 1e-6);
        let norm_result = self.compute_engine.compute_rmsnorm(norm_params2)?;

        // Gate mechanism: x * sigmoid(gate(x))
        let gate_proj = self.project_to_ffn(&norm_result.output, ffn_gate)?;
        let up_proj = self.project_to_ffn(&norm_result.output, ffn_up)?;

        let gated = self.apply_silu_gate(&up_proj, &gate_proj)?;
        let ffn_result = self
            .compute_engine
            .compute_element_mul(&gated, &gate_proj)?;

        // Down projection
        let ffn_out = self.project_from_ffn(&ffn_result.output, ffn_down)?;
        let ffn_time = ffn_start.elapsed().as_secs_f32() * 1000.0;

        // Final residual connection
        let output = self.add_residual(&residual, &ffn_out);

        let total_time = total_start.elapsed().as_secs_f32() * 1000.0;

        Ok(GPUInferenceResult {
            output,
            attention_time_ms: attention_time,
            ffn_time_ms: ffn_time,
            total_time_ms: total_time,
            used_gpu: self.config.gpu_enabled,
        })
    }

    /// Project input to attention query dimension
    fn project_to_attention(&self, input: &[f32], weight: &[f32]) -> MinervaResult<Vec<f32>> {
        let seq_len = input.len() / self.config.hidden_dim;
        let output_dim = self.config.num_heads * self.config.head_dim;

        if weight.len() != self.config.hidden_dim * output_dim {
            return Err(MinervaError::InferenceError(
                "Weight dimension mismatch for attention projection".to_string(),
            ));
        }

        let params = MatmulParams::new(
            input.to_vec(),
            weight.to_vec(),
            seq_len,
            self.config.hidden_dim,
        );

        let result = self.compute_engine.compute_matmul(params)?;
        Ok(result.output)
    }

    /// Project attention output back to hidden dimension
    fn project_output(&self, input: &[f32], weight: &[f32]) -> MinervaResult<Vec<f32>> {
        let seq_len = input.len() / (self.config.num_heads * self.config.head_dim);
        let input_dim = self.config.num_heads * self.config.head_dim;

        if weight.len() != input_dim * self.config.hidden_dim {
            return Err(MinervaError::InferenceError(
                "Weight dimension mismatch for output projection".to_string(),
            ));
        }

        let params = MatmulParams::new(input.to_vec(), weight.to_vec(), seq_len, input_dim);

        let result = self.compute_engine.compute_matmul(params)?;
        Ok(result.output)
    }

    /// Project to FFN hidden dimension
    fn project_to_ffn(&self, input: &[f32], weight: &[f32]) -> MinervaResult<Vec<f32>> {
        let seq_len = input.len() / self.config.hidden_dim;

        if weight.len() != self.config.hidden_dim * self.config.intermediate_dim {
            return Err(MinervaError::InferenceError(
                "Weight dimension mismatch for FFN up projection".to_string(),
            ));
        }

        let params = MatmulParams::new(
            input.to_vec(),
            weight.to_vec(),
            seq_len,
            self.config.hidden_dim,
        );

        let result = self.compute_engine.compute_matmul(params)?;
        Ok(result.output)
    }

    /// Project from FFN hidden dimension back
    fn project_from_ffn(&self, input: &[f32], weight: &[f32]) -> MinervaResult<Vec<f32>> {
        let seq_len = input.len() / self.config.intermediate_dim;

        if weight.len() != self.config.intermediate_dim * self.config.hidden_dim {
            return Err(MinervaError::InferenceError(
                "Weight dimension mismatch for FFN down projection".to_string(),
            ));
        }

        let params = MatmulParams::new(
            input.to_vec(),
            weight.to_vec(),
            seq_len,
            self.config.intermediate_dim,
        );

        let result = self.compute_engine.compute_matmul(params)?;
        Ok(result.output)
    }

    /// Apply SiLU gating to FFN output
    fn apply_silu_gate(&self, x: &[f32], gate: &[f32]) -> MinervaResult<Vec<f32>> {
        if x.len() != gate.len() {
            return Err(MinervaError::InferenceError(
                "SiLU gate dimension mismatch".to_string(),
            ));
        }

        let silu_gate: Vec<f32> = gate.iter().map(|&g| g / (1.0 + (-g).exp())).collect();

        Ok(silu_gate)
    }

    /// Add residual connection
    fn add_residual(&self, input: &[f32], output: &[f32]) -> Vec<f32> {
        input
            .iter()
            .zip(output.iter())
            .map(|(a, b)| a + b)
            .collect()
    }

    /// Check if GPU is enabled
    pub fn is_gpu_enabled(&self) -> bool {
        self.config.gpu_enabled
    }

    /// Get configuration
    pub fn config(&self) -> &GPUInferenceConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_inference_config_default() {
        let config = GPUInferenceConfig::default();
        assert!(config.gpu_enabled);
        assert!(config.use_simulation);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.head_dim, 64);
    }

    #[test]
    fn test_gpu_llama_inference_creation() {
        let config = GPUInferenceConfig::default();
        let inference = GPULlamaInference::new(config).unwrap();
        assert!(inference.is_gpu_enabled());
    }

    #[test]
    fn test_gpu_llama_inference_simulated() {
        let inference = GPULlamaInference::simulated().unwrap();
        assert!(inference.is_gpu_enabled());
    }

    #[test]
    fn test_forward_block_basic() {
        let config = GPUInferenceConfig {
            gpu_enabled: true,
            use_simulation: true,
            num_heads: 2,
            head_dim: 4,
            hidden_dim: 8,
            intermediate_dim: 16,
        };
        let inference = GPULlamaInference::new(config).unwrap();

        let batch_size = 1;
        let _seq_len = 1;
        let hidden_dim = 8;
        let num_heads = 2;
        let head_dim = 4;
        let intermediate_dim = 16;

        let input = vec![0.5; batch_size * hidden_dim];
        let q_weight = vec![0.1; hidden_dim * num_heads * head_dim];
        let k_weight = vec![0.1; hidden_dim * num_heads * head_dim];
        let v_weight = vec![0.1; hidden_dim * num_heads * head_dim];
        let o_weight = vec![0.1; num_heads * head_dim * hidden_dim];
        let ffn_up = vec![0.1; hidden_dim * intermediate_dim];
        let ffn_down = vec![0.1; intermediate_dim * hidden_dim];
        let ffn_gate = vec![0.1; hidden_dim * intermediate_dim];
        let norm_weight = vec![1.0; hidden_dim];

        let result = inference.forward_block(
            &input,
            &q_weight,
            &k_weight,
            &v_weight,
            &o_weight,
            &ffn_up,
            &ffn_down,
            &ffn_gate,
            &norm_weight,
        );

        assert!(result.is_ok());
        let res = result.unwrap();
        assert_eq!(res.output.len(), hidden_dim);
        assert!(res.total_time_ms >= 0.0);
    }

    #[test]
    fn test_forward_block_invalid_input() {
        let config = GPUInferenceConfig::default();
        let inference = GPULlamaInference::new(config).unwrap();

        let input = vec![0.5; 256]; // Wrong size
        let weights = vec![0.1; 512 * 512];

        let result = inference.forward_block(
            &input, &weights, &weights, &weights, &weights, &weights, &weights, &weights, &weights,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_attention_projection() {
        let config = GPUInferenceConfig {
            num_heads: 2,
            head_dim: 4,
            hidden_dim: 8,
            ..Default::default()
        };
        let inference = GPULlamaInference::new(config).unwrap();

        let input = vec![0.5; 8]; // 1 token, hidden_dim=8
        let weight = vec![0.1; 8 * 8]; // 8 * (2 heads * 4 dim) = 8 * 8

        let result = inference.project_to_attention(&input, &weight);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 8);
    }

    #[test]
    fn test_ffn_projections() {
        let config = GPUInferenceConfig {
            hidden_dim: 8,
            intermediate_dim: 16,
            ..Default::default()
        };
        let inference = GPULlamaInference::new(config).unwrap();

        let input = vec![0.5; 8];
        let up_weight = vec![0.1; 8 * 16];
        let down_weight = vec![0.1; 16 * 8];

        let up_result = inference.project_to_ffn(&input, &up_weight);
        assert!(up_result.is_ok());
        assert_eq!(up_result.unwrap().len(), 16);

        let ffn_output = vec![0.5; 16];
        let down_result = inference.project_from_ffn(&ffn_output, &down_weight);
        assert!(down_result.is_ok());
        assert_eq!(down_result.unwrap().len(), 8);
    }

    #[test]
    fn test_residual_connection() {
        let config = GPUInferenceConfig::default();
        let inference = GPULlamaInference::new(config).unwrap();

        let input = vec![1.0; 10];
        let output = vec![0.5; 10];

        let result = inference.add_residual(&input, &output);
        assert_eq!(result.len(), 10);
        for &val in &result {
            assert!((val - 1.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_silu_gate() {
        let config = GPUInferenceConfig::default();
        let inference = GPULlamaInference::new(config).unwrap();

        let x = vec![1.0, 2.0, 3.0];
        let gate = vec![1.0, 2.0, 3.0];

        let result = inference.apply_silu_gate(&x, &gate);
        assert!(result.is_ok());
        let silu = result.unwrap();
        assert_eq!(silu.len(), 3);

        // SiLU: x / (1 + exp(-x))
        for (i, &val) in silu.iter().enumerate() {
            let expected = gate[i] / (1.0 + (-gate[i]).exp());
            assert!((val - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_silu_gate_mismatch() {
        let config = GPUInferenceConfig::default();
        let inference = GPULlamaInference::new(config).unwrap();

        let x = vec![1.0, 2.0];
        let gate = vec![1.0, 2.0, 3.0];

        let result = inference.apply_silu_gate(&x, &gate);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_getter() {
        let config = GPUInferenceConfig {
            num_heads: 4,
            head_dim: 8,
            ..Default::default()
        };
        let inference = GPULlamaInference::new(config.clone()).unwrap();
        assert_eq!(inference.config().num_heads, 4);
        assert_eq!(inference.config().head_dim, 8);
    }
}
