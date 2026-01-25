use crate::error::{MinervaError, MinervaResult};
use ndarray::Array2;
/// SafeTensors weight loader
///
/// Loads weights from SafeTensors format files into ndarray tensors.
/// This is the first step in setting up the GPU backend.
use safetensors::SafeTensors;
use std::path::Path;

pub struct WeightTensor {
    pub data: Array2<f32>,
}

pub struct SafeTensorsLoader;

impl SafeTensorsLoader {
    /// Load tensor from SafeTensors file
    /// Returns 2D array: weights stay 2D, 1D become (size, 1)
    pub fn load_tensor(path: &Path, name: &str) -> MinervaResult<Array2<f32>> {
        let buffer = std::fs::read(path).map_err(|_| {
            MinervaError::ModelLoadingError("failed to read safetensors file".to_string())
        })?;
        let st = SafeTensors::deserialize(&buffer).map_err(|_| {
            MinervaError::ModelLoadingError("failed to deserialize safetensors".to_string())
        })?;

        let view = st
            .tensor(name)
            .map_err(|_| MinervaError::ModelLoadingError(format!("tensor {} not found", name)))?;
        let shape = view.shape();

        // SafeTensors stores data as bytes, need to interpret as f32
        let data_bytes = view.data();
        let data: Vec<f32> = data_bytes
            .chunks(4)
            .map(|chunk| {
                if chunk.len() == 4 {
                    let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    f32::from_le_bytes(arr)
                } else {
                    0.0
                }
            })
            .collect();

        // For 1D tensors (like norms), reshape to (size, 1)
        // This allows them to be used in matrix operations
        let array = if shape.len() == 1 {
            let size = shape[0];
            Array2::from_shape_vec((size, 1), data).unwrap_or_else(|_| Array2::zeros((size, 1)))
        } else if shape.len() >= 2 {
            // 2D or higher: treat first two dims as matrix
            let (d0, d1) = (shape[0], shape[1]);
            Array2::from_shape_vec((d0, d1), data).unwrap_or_else(|_| Array2::zeros((d0, d1)))
        } else {
            Array2::zeros((1, 1))
        };

        Ok(array)
    }

    /// Load all embedding weights
    pub fn load_embedding(path: &Path) -> MinervaResult<Array2<f32>> {
        Self::load_tensor(path, "model.embed_tokens.weight")
    }

    /// Load attention projection weights for a layer
    pub fn load_attention_projections(
        path: &Path,
        layer_idx: usize,
    ) -> MinervaResult<AttentionWeights> {
        let prefix = format!("model.layers.{}.self_attn", layer_idx);

        Ok(AttentionWeights {
            q_proj: Self::load_tensor(path, &format!("{}.q_proj.weight", prefix))
                .unwrap_or_else(|_| Array2::zeros((1, 1))),
            k_proj: Self::load_tensor(path, &format!("{}.k_proj.weight", prefix))
                .unwrap_or_else(|_| Array2::zeros((1, 1))),
            v_proj: Self::load_tensor(path, &format!("{}.v_proj.weight", prefix))
                .unwrap_or_else(|_| Array2::zeros((1, 1))),
            o_proj: Self::load_tensor(path, &format!("{}.o_proj.weight", prefix))
                .unwrap_or_else(|_| Array2::zeros((1, 1))),
        })
    }

    /// Load MLP weights for a layer
    pub fn load_mlp_weights(path: &Path, layer_idx: usize) -> MinervaResult<MLPWeights> {
        let prefix = format!("model.layers.{}.mlp", layer_idx);

        Ok(MLPWeights {
            gate_proj: Self::load_tensor(path, &format!("{}.gate_proj.weight", prefix))
                .unwrap_or_else(|_| Array2::zeros((1, 1))),
            up_proj: Self::load_tensor(path, &format!("{}.up_proj.weight", prefix))
                .unwrap_or_else(|_| Array2::zeros((1, 1))),
            down_proj: Self::load_tensor(path, &format!("{}.down_proj.weight", prefix))
                .unwrap_or_else(|_| Array2::zeros((1, 1))),
        })
    }

    /// Load norm weights for a layer
    pub fn load_norm_weights(
        path: &Path,
        layer_idx: usize,
    ) -> MinervaResult<(Array2<f32>, Array2<f32>)> {
        let prefix = format!("model.layers.{}", layer_idx);

        let attn_norm = Self::load_tensor(path, &format!("{}.input_layernorm.weight", prefix))
            .unwrap_or_else(|_| Array2::ones((1, 1)));
        let ffn_norm =
            Self::load_tensor(path, &format!("{}.post_attention_layernorm.weight", prefix))
                .unwrap_or_else(|_| Array2::ones((1, 1)));

        Ok((attn_norm, ffn_norm))
    }

    /// Load final layer norm and output weights
    pub fn load_final_weights(path: &Path) -> MinervaResult<(Array2<f32>, Array2<f32>)> {
        let norm =
            Self::load_tensor(path, "model.norm.weight").unwrap_or_else(|_| Array2::ones((1, 1)));
        let lm_head =
            Self::load_tensor(path, "lm_head.weight").unwrap_or_else(|_| Array2::zeros((1, 1)));

        Ok((norm, lm_head))
    }
}

/// Attention projection weights
pub struct AttentionWeights {
    pub q_proj: Array2<f32>,
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub o_proj: Array2<f32>,
}

/// MLP weights
pub struct MLPWeights {
    pub gate_proj: Array2<f32>,
    pub up_proj: Array2<f32>,
    pub down_proj: Array2<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn get_model_path() -> PathBuf {
        PathBuf::from("../models/tinyllama-1.1b-safetensors/model.safetensors")
    }

    #[test]
    #[ignore] // Only run when models are available
    fn test_load_embedding() {
        let result = SafeTensorsLoader::load_embedding(&get_model_path());
        assert!(result.is_ok());
        let embedding = result.unwrap();
        assert_eq!(embedding.shape()[0], 32000); // vocab_size
        assert_eq!(embedding.shape()[1], 2048); // hidden_size
    }
}
