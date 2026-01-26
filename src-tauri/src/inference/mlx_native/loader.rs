use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use super::config::GPTOSSConfig;
use crate::error::MinervaResult;

#[path = "loader_helpers.rs"]
mod helpers;

use helpers::{extract_tensor_1d, extract_tensor_2d, load_safetensors_files};

/// Layer weights for a single transformer layer
#[derive(Debug, Clone)]
pub struct MLXLayerWeights {
    pub attn_q: Array2<f32>,
    pub attn_k: Array2<f32>,
    pub attn_v: Array2<f32>,
    pub attn_out: Array2<f32>,
    pub mlp_gate: Array2<f32>,
    pub mlp_up: Array2<f32>,
    pub mlp_down: Array2<f32>,
    pub norm_attn: Array1<f32>,
    pub norm_mlp: Array1<f32>,
}

/// Complete MLX model with all weights
#[derive(Debug, Clone)]
pub struct MLXModel {
    pub embedding: Array2<f32>,
    pub lm_head: Array2<f32>,
    pub layers: Vec<MLXLayerWeights>,
    pub norm_final: Array1<f32>,
    pub config: GPTOSSConfig,
}

impl MLXModel {
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn embedding_dim(&self) -> usize {
        self.config.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
}

/// Load a single layer's weights
fn load_layer(tensors: &HashMap<String, Vec<u8>>, idx: usize) -> MinervaResult<MLXLayerWeights> {
    let prefix = format!("model.layers.{}", idx);
    Ok(MLXLayerWeights {
        attn_q: extract_tensor_2d(tensors, &format!("{}.self_attn.q_proj.weight", prefix))?,
        attn_k: extract_tensor_2d(tensors, &format!("{}.self_attn.k_proj.weight", prefix))?,
        attn_v: extract_tensor_2d(tensors, &format!("{}.self_attn.v_proj.weight", prefix))?,
        attn_out: extract_tensor_2d(tensors, &format!("{}.self_attn.o_proj.weight", prefix))?,
        mlp_gate: extract_tensor_2d(tensors, &format!("{}.mlp.gate_proj.weight", prefix))?,
        mlp_up: extract_tensor_2d(tensors, &format!("{}.mlp.up_proj.weight", prefix))?,
        mlp_down: extract_tensor_2d(tensors, &format!("{}.mlp.down_proj.weight", prefix))?,
        norm_attn: extract_tensor_1d(tensors, &format!("{}.input_layernorm.weight", prefix))?,
        norm_mlp: extract_tensor_1d(
            tensors,
            &format!("{}.post_attention_layernorm.weight", prefix),
        )?,
    })
}

/// Load MLX model from SafeTensors files
pub fn load_mlx_model(path: &Path) -> MinervaResult<MLXModel> {
    let start = Instant::now();
    let tensors = load_safetensors_files(path)?;

    let embedding = extract_tensor_2d(&tensors, "model.embed_tokens.weight")?;
    let lm_head = extract_tensor_2d(&tensors, "lm_head.weight")?;
    let norm_final = extract_tensor_1d(&tensors, "model.norm.weight")?;

    let mut layers = Vec::new();
    for layer_idx in 0..24 {
        layers.push(load_layer(&tensors, layer_idx)?);
    }

    let elapsed = start.elapsed();
    eprintln!(
        "Loaded MLX model in {:.2}ms",
        elapsed.as_secs_f64() * 1000.0
    );

    Ok(MLXModel {
        embedding,
        lm_head,
        layers,
        norm_final,
        config: GPTOSSConfig::default(),
    })
}

#[cfg(test)]
#[path = "loader_tests.rs"]
mod tests;
