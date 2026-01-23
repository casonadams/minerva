use super::metal_gpu::{GPUMemoryPool, MetalDevice, MetalDeviceInfo};
/// GPU Computation Engine - Phase 6 Step 4
///
/// This module integrates Metal GPU operations with LLaMA inference,
/// providing high-level GPU compute capabilities for:
/// - Attention computation on GPU
/// - Feed-forward network computation on GPU
/// - Matrix operations with GPU acceleration
///
/// Automatically falls back to CPU for testing and unsupported operations.
use crate::error::{MinervaError, MinervaResult};
use std::sync::Arc;

/// GPU Computation Configuration
#[derive(Debug, Clone)]
pub struct GPUComputeConfig {
    /// Enable GPU acceleration (set to false to force CPU)
    pub enabled: bool,
    /// GPU pool size in bytes (default: 4GB)
    pub pool_size: usize,
    /// Use simulated GPU (for testing)
    pub use_simulation: bool,
}

impl Default for GPUComputeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pool_size: 4 * 1024 * 1024 * 1024, // 4GB
            use_simulation: true,
        }
    }
}

/// GPU Computation Result with timing
#[derive(Debug, Clone)]
pub struct ComputeResult {
    /// Output buffer data
    pub output: Vec<f32>,
    /// Execution time in milliseconds
    pub execution_time_ms: f32,
    /// Whether computation used GPU
    pub used_gpu: bool,
}

/// Parameters for matrix multiplication
#[derive(Debug, Clone)]
pub struct MatmulParams {
    /// Matrix A dimensions
    pub a_rows: usize,
    pub a_cols: usize,
    /// Output columns
    pub b_cols: usize,
}

/// Parameters for attention computation
#[derive(Debug, Clone)]
pub struct AttentionParams {
    /// Number of attention heads
    pub heads: usize,
    /// Dimension per head
    pub head_dim: usize,
}

/// Parameters for RMSNorm computation
#[derive(Debug, Clone)]
pub struct RmsNormParams {
    /// Normalization epsilon
    pub eps: f32,
}

/// GPU Computation Engine
pub struct GPUComputeEngine {
    /// Metal device for GPU operations
    device: Arc<MetalDevice>,
    /// GPU memory pool
    #[allow(dead_code)]
    memory_pool: Arc<GPUMemoryPool>,
    /// Configuration
    config: GPUComputeConfig,
}

impl GPUComputeEngine {
    /// Create new GPU compute engine
    pub fn new(config: GPUComputeConfig) -> MinervaResult<Self> {
        let device = Arc::new(MetalDevice::simulated());
        // Note: Real Metal device creation would occur here if available
        // Currently using simulated mode for cross-platform compatibility

        let memory_pool = Arc::new(GPUMemoryPool::new(config.pool_size));

        Ok(Self {
            device,
            memory_pool,
            config,
        })
    }

    /// Create simulated GPU engine (for testing)
    pub fn simulated() -> MinervaResult<Self> {
        let config = GPUComputeConfig {
            enabled: true,
            use_simulation: true,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Check if GPU is enabled
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get device info
    pub fn device_info(&self) -> MetalDeviceInfo {
        self.device.info().clone()
    }

    /// Execute matrix multiplication on GPU
    pub fn compute_matmul(
        &self,
        a: &[f32],
        b: &[f32],
        params: MatmulParams,
    ) -> MinervaResult<ComputeResult> {
        let start = std::time::Instant::now();

        if !self.config.enabled {
            return self.cpu_matmul(a, b, params);
        }

        // For now, fall back to CPU for actual computation
        // In production, this would use GPU kernels
        let output = self.cpu_matmul_impl(a, b, &params);
        let elapsed = start.elapsed();

        Ok(ComputeResult {
            output,
            execution_time_ms: elapsed.as_secs_f32() * 1000.0,
            used_gpu: false, // Simulated for now
        })
    }

    /// Execute attention computation on GPU
    pub fn compute_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        params: AttentionParams,
    ) -> MinervaResult<ComputeResult> {
        let start = std::time::Instant::now();

        if !self.config.enabled {
            return self.cpu_attention(q, k, v, params);
        }

        // Validate input shapes
        let seq_len = q.len() / (params.heads * params.head_dim);
        if q.len() != params.heads * params.head_dim * seq_len {
            return Err(MinervaError::InferenceError(
                "Invalid attention input shape".to_string(),
            ));
        }

        // For now, fall back to CPU
        let output = self.cpu_attention_impl(q, k, v, &params, seq_len);
        let elapsed = start.elapsed();

        Ok(ComputeResult {
            output,
            execution_time_ms: elapsed.as_secs_f32() * 1000.0,
            used_gpu: false, // Simulated for now
        })
    }

    /// Execute element-wise operations on GPU
    pub fn compute_element_mul(&self, a: &[f32], b: &[f32]) -> MinervaResult<ComputeResult> {
        let start = std::time::Instant::now();

        if a.len() != b.len() {
            return Err(MinervaError::InferenceError(
                "Element multiplication requires same-sized inputs".to_string(),
            ));
        }

        if !self.config.enabled {
            let output = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
            return Ok(ComputeResult {
                output,
                execution_time_ms: 0.0,
                used_gpu: false,
            });
        }

        let output = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        let elapsed = start.elapsed();

        Ok(ComputeResult {
            output,
            execution_time_ms: elapsed.as_secs_f32() * 1000.0,
            used_gpu: false, // Simulated for now
        })
    }

    /// Execute RMSNorm on GPU
    pub fn compute_rmsnorm(
        &self,
        x: &[f32],
        weight: &[f32],
        params: RmsNormParams,
    ) -> MinervaResult<ComputeResult> {
        let start = std::time::Instant::now();

        if x.len() != weight.len() {
            return Err(MinervaError::InferenceError(
                "RMSNorm weight mismatch".to_string(),
            ));
        }

        if !self.config.enabled {
            let output = self.cpu_rmsnorm_impl(x, weight, params.eps);
            return Ok(ComputeResult {
                output,
                execution_time_ms: 0.0,
                used_gpu: false,
            });
        }

        let output = self.cpu_rmsnorm_impl(x, weight, params.eps);
        let elapsed = start.elapsed();

        Ok(ComputeResult {
            output,
            execution_time_ms: elapsed.as_secs_f32() * 1000.0,
            used_gpu: false, // Simulated for now
        })
    }

    /// CPU fallback for matrix multiplication
    fn cpu_matmul(
        &self,
        a: &[f32],
        b: &[f32],
        params: MatmulParams,
    ) -> MinervaResult<ComputeResult> {
        let start = std::time::Instant::now();
        let output = self.cpu_matmul_impl(a, b, &params);
        let elapsed = start.elapsed();

        Ok(ComputeResult {
            output,
            execution_time_ms: elapsed.as_secs_f32() * 1000.0,
            used_gpu: false,
        })
    }

    /// CPU implementation of matrix multiplication
    fn cpu_matmul_impl(&self, a: &[f32], b: &[f32], params: &MatmulParams) -> Vec<f32> {
        let mut c = vec![0.0; params.a_rows * params.b_cols];

        for i in 0..params.a_rows {
            for j in 0..params.b_cols {
                let mut sum = 0.0;
                for k in 0..params.a_cols {
                    sum += a[i * params.a_cols + k] * b[k * params.b_cols + j];
                }
                c[i * params.b_cols + j] = sum;
            }
        }

        c
    }

    /// CPU fallback for attention
    fn cpu_attention(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        params: AttentionParams,
    ) -> MinervaResult<ComputeResult> {
        let start = std::time::Instant::now();
        let seq_len = q.len() / (params.heads * params.head_dim);
        let output = self.cpu_attention_impl(q, k, v, &params, seq_len);
        let elapsed = start.elapsed();

        Ok(ComputeResult {
            output,
            execution_time_ms: elapsed.as_secs_f32() * 1000.0,
            used_gpu: false,
        })
    }

    /// CPU implementation of attention
    fn cpu_attention_impl(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        params: &AttentionParams,
        seq_len: usize,
    ) -> Vec<f32> {
        let mut output = vec![0.0; q.len()];
        let scale = 1.0 / (params.head_dim as f32).sqrt();

        for h in 0..params.heads {
            for i in 0..seq_len {
                // Query-key scores
                let mut max_score = f32::NEG_INFINITY;
                let mut scores = vec![0.0; seq_len];

                for j in 0..seq_len {
                    let mut score = 0.0;
                    for d in 0..params.head_dim {
                        let q_idx = h * seq_len * params.head_dim + i * params.head_dim + d;
                        let k_idx = h * seq_len * params.head_dim + j * params.head_dim + d;
                        score += q[q_idx] * k[k_idx];
                    }
                    score *= scale;
                    if score > max_score {
                        max_score = score;
                    }
                    scores[j] = score;
                }

                // Softmax
                let mut exp_sum = 0.0;
                for score in &mut scores {
                    let exp_val = (*score - max_score).exp();
                    *score = exp_val;
                    exp_sum += exp_val;
                }

                for score in &mut scores {
                    *score /= exp_sum;
                }

                // Aggregate values
                for d in 0..params.head_dim {
                    let mut val = 0.0;
                    for (j, &score) in scores.iter().enumerate() {
                        let v_idx = h * seq_len * params.head_dim + j * params.head_dim + d;
                        val += score * v[v_idx];
                    }
                    let out_idx = h * seq_len * params.head_dim + i * params.head_dim + d;
                    output[out_idx] = val;
                }
            }
        }

        output
    }

    /// CPU implementation of RMSNorm
    fn cpu_rmsnorm_impl(&self, x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
        let mut output = vec![0.0; x.len()];
        let dim = x.len();

        // Compute RMS
        let mut sum_sq = 0.0;
        for &v in x {
            sum_sq += v * v;
        }
        let rms = (sum_sq / dim as f32 + eps).sqrt();

        // Normalize and scale
        for i in 0..dim {
            output[i] = (x[i] / rms) * weight[i];
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_compute_config_default() {
        let config = GPUComputeConfig::default();
        assert!(config.enabled);
        assert!(config.use_simulation);
        assert_eq!(config.pool_size, 4 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_gpu_compute_engine_creation() {
        let config = GPUComputeConfig {
            enabled: true,
            use_simulation: true,
            pool_size: 1024 * 1024,
        };
        let engine = GPUComputeEngine::new(config).unwrap();
        assert!(engine.is_enabled());
    }

    #[test]
    fn test_gpu_compute_engine_simulated() {
        let engine = GPUComputeEngine::simulated().unwrap();
        assert!(engine.is_enabled());
        let info = engine.device_info();
        assert!(info.is_simulated);
    }

    #[test]
    fn test_cpu_matmul() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0, 6.0, 7.0, 8.0]; // 2x2

        let params = MatmulParams {
            a_rows: 2,
            a_cols: 2,
            b_cols: 2,
        };
        let result = engine.compute_matmul(&a, &b, params).unwrap();

        // Verify result shape
        assert_eq!(result.output.len(), 4);

        // Verify computation: [1,2]*[5,7] = 1*5 + 2*7 = 19
        assert!((result.output[0] - 19.0).abs() < 0.01);
        // [1,2]*[6,8] = 1*6 + 2*8 = 22
        assert!((result.output[1] - 22.0).abs() < 0.01);
    }

    #[test]
    fn test_element_mul() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![2.0, 3.0, 4.0];

        let result = engine.compute_element_mul(&a, &b).unwrap();
        assert_eq!(result.output, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_element_mul_size_mismatch() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];

        let result = engine.compute_element_mul(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_rmsnorm() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];

        let params = RmsNormParams { eps: 1e-6 };
        let result = engine.compute_rmsnorm(&x, &weight, params).unwrap();
        assert_eq!(result.output.len(), 4);

        // RMS should normalize the values
        let rms_sq: f32 = result.output.iter().map(|x| x * x).sum();
        let rms = (rms_sq / 4.0).sqrt();

        // After RMSNorm with weight 1.0, should be scaled normalized
        assert!(rms > 0.0 && rms < 2.0);
    }

    #[test]
    fn test_rmsnorm_weight_mismatch() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let weight = vec![1.0, 1.0];

        let params = RmsNormParams { eps: 1e-6 };
        let result = engine.compute_rmsnorm(&x, &weight, params);
        assert!(result.is_err());
    }

    #[test]
    fn test_attention_simple() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let heads = 1;
        let head_dim = 2;

        // Q, K, V: 1 head, 2 positions, 2 dims each
        let q = vec![1.0, 0.0, 0.0, 1.0]; // 2 queries
        let k = vec![1.0, 0.0, 0.0, 1.0]; // 2 keys
        let v = vec![2.0, 3.0, 4.0, 5.0]; // 2 values

        let params = AttentionParams { heads, head_dim };
        let result = engine.compute_attention(&q, &k, &v, params).unwrap();
        assert_eq!(result.output.len(), 4);
    }

    #[test]
    fn test_attention_invalid_shape() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let q = vec![1.0, 2.0, 3.0];
        let k = vec![1.0, 2.0];
        let v = vec![1.0, 2.0];

        let params = AttentionParams {
            heads: 2,
            head_dim: 2,
        };
        let result = engine.compute_attention(&q, &k, &v, params);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_disabled() {
        let config = GPUComputeConfig {
            enabled: false,
            use_simulation: false,
            ..Default::default()
        };
        let engine = GPUComputeEngine::new(config).unwrap();
        assert!(!engine.is_enabled());
    }

    #[test]
    fn test_matmul_execution_time() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let a = vec![1.0; 100];
        let b = vec![1.0; 100];

        let params = MatmulParams {
            a_rows: 10,
            a_cols: 10,
            b_cols: 10,
        };
        let result = engine.compute_matmul(&a, &b, params).unwrap();
        assert!(result.execution_time_ms >= 0.0);
    }

    #[test]
    fn test_attention_with_multiple_heads() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let heads = 4;
        let head_dim = 8;
        let seq_len = 2;

        let size = heads * seq_len * head_dim;
        let q = vec![0.5; size];
        let k = vec![0.5; size];
        let v = vec![1.0; size];

        let params = AttentionParams { heads, head_dim };
        let result = engine.compute_attention(&q, &k, &v, params).unwrap();
        assert_eq!(result.output.len(), size);
    }

    #[test]
    fn test_element_mul_computation() {
        let engine = GPUComputeEngine::simulated().unwrap();
        let a = vec![2.5, 3.5, 1.5];
        let b = vec![2.0, 2.0, 2.0];

        let result = engine.compute_element_mul(&a, &b).unwrap();
        let expected = [5.0, 7.0, 3.0];

        for (i, &val) in result.output.iter().enumerate() {
            assert!((val - expected[i]).abs() < 0.01);
        }
    }
}
