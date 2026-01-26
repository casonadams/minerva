#[cfg(test)]
mod tests {
    use crate::inference::mlx_native::kv_quantization::QuantizedKVCache;
    use crate::inference::mlx_native::{ArrayShape, MLXArray};

    #[test]
    fn test_quantize_and_dequant() {
        let k_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let k = MLXArray::new_cpu(k_data.clone(), ArrayShape::Shape1D(5));
        let v_data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let v = MLXArray::new_cpu(v_data, ArrayShape::Shape1D(5));
        let cache = QuantizedKVCache::quantize(&k, &v);
        let k_recon = cache.dequant_k(0, 5);
        let mut max_error = 0.0f32;
        for i in 0..5 {
            max_error = max_error.max((k_data[i] - k_recon[i]).abs());
        }
        assert!(max_error < (5.0 - 1.0) * 0.1);
    }

    #[test]
    fn test_compression_ratio() {
        let k_data = vec![1.0; 1000];
        let k = MLXArray::new_cpu(k_data, ArrayShape::Shape1D(1000));
        let v_data = vec![1.0; 1000];
        let v = MLXArray::new_cpu(v_data, ArrayShape::Shape1D(1000));
        let cache = QuantizedKVCache::quantize(&k, &v);
        assert!(cache.compression_ratio() > 3.0);
    }
}
