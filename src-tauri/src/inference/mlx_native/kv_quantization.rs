use super::kv_quantization_helpers::{dequantize_range, quantize_tensor};
use super::unified_memory::{ArrayShape, MLXArray};

/// Quantized KV cache for memory efficiency
/// Reduces memory usage by 8x (float32 -> uint8)
/// Typical usage: 71GB → 9GB for 128K context
#[derive(Clone)]
pub struct QuantizedKVCache {
    k_quant: Vec<u8>,
    v_quant: Vec<u8>,
    k_scales: Vec<f32>,
    k_mins: Vec<f32>,
    v_scales: Vec<f32>,
    v_mins: Vec<f32>,
    shape: (usize, usize),
    block_size: usize,
}

impl QuantizedKVCache {
    const DEFAULT_BLOCK_SIZE: usize = 32;

    pub fn new() -> Self {
        QuantizedKVCache {
            k_quant: Vec::new(),
            v_quant: Vec::new(),
            k_scales: Vec::new(),
            k_mins: Vec::new(),
            v_scales: Vec::new(),
            v_mins: Vec::new(),
            shape: (0, 0),
            block_size: Self::DEFAULT_BLOCK_SIZE,
        }
    }

    /// Quantize float32 K, V tensors to uint8
    pub fn quantize(k: &MLXArray, v: &MLXArray) -> Self {
        let k_data = k.data();
        let v_data = v.data();

        let block_size = Self::DEFAULT_BLOCK_SIZE;
        let num_blocks = (k_data.len() + block_size - 1) / block_size;

        let (k_quant, k_scales, k_mins) = quantize_tensor(&k_data, block_size, num_blocks);
        let (v_quant, v_scales, v_mins) = quantize_tensor(&v_data, block_size, num_blocks);

        let shape = match k.shape() {
            ArrayShape::Shape2D(m, n) => (m, n),
            _ => (k_data.len(), 1),
        };

        QuantizedKVCache {
            k_quant,
            v_quant,
            k_scales,
            k_mins,
            v_scales,
            v_mins,
            shape,
            block_size,
        }
    }

    pub fn dequant_k(&self, start: usize, end: usize) -> Vec<f32> {
        dequantize_range(
            &self.k_quant,
            &self.k_scales,
            &self.k_mins,
            self.block_size,
            start,
            end,
        )
    }

    pub fn dequant_v(&self, start: usize, end: usize) -> Vec<f32> {
        dequantize_range(
            &self.v_quant,
            &self.v_scales,
            &self.v_mins,
            self.block_size,
            start,
            end,
        )
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn memory_usage(&self) -> usize {
        self.k_quant.len()
            + self.v_quant.len()
            + (self.k_scales.len() + self.k_mins.len() + self.v_scales.len() + self.v_mins.len())
                * 4
    }

    pub fn original_memory_usage(&self) -> usize {
        (self.k_quant.len() + self.v_quant.len()) * 4
    }

    pub fn compression_ratio(&self) -> f32 {
        self.original_memory_usage() as f32 / self.memory_usage() as f32
    }

    pub fn k_quant(&self) -> &[u8] {
        &self.k_quant
    }

    pub fn v_quant(&self) -> &[u8] {
        &self.v_quant
    }

    pub fn k_scales(&self) -> &[f32] {
        &self.k_scales
    }

    pub fn v_scales(&self) -> &[f32] {
        &self.v_scales
    }

    pub fn total_elements(&self) -> usize {
        self.k_quant.len()
    }
}
