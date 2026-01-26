/// Find min and max in a slice
pub(crate) fn find_min_max(data: &[f32]) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 1.0);
    }

    let mut min = data[0];
    let mut max = data[0];

    for &val in data {
        if val < min {
            min = val;
        }
        if val > max {
            max = val;
        }
    }

    (min, max)
}

/// Dequantize a range of values
pub(crate) fn dequantize_range(
    quant_data: &[u8],
    scales: &[f32],
    mins: &[f32],
    block_size: usize,
    start: usize,
    end: usize,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(end - start);

    for i in start..end {
        let block_idx = i / block_size;
        let scale = scales.get(block_idx).copied().unwrap_or(1.0);
        let min_val = mins.get(block_idx).copied().unwrap_or(0.0);
        let quant_val = quant_data[i];
        result.push(quant_val as f32 * scale + min_val);
    }

    result
}

/// Quantize a tensor using block-wise int8 quantization
pub(crate) fn quantize_tensor(
    data: &[f32],
    block_size: usize,
    num_blocks: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut quant = vec![0u8; data.len()];
    let mut scales = vec![0.0f32; num_blocks];
    let mut mins = vec![0.0f32; num_blocks];

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = (start + block_size).min(data.len());
        let block = &data[start..end];

        let (min_val, max_val) = find_min_max(block);
        let scale = (max_val - min_val) / 255.0;

        scales[block_idx] = scale;
        mins[block_idx] = min_val;

        for i in start..end {
            let normalized = (data[i] - min_val) / scale;
            quant[i] = normalized.clamp(0.0, 255.0) as u8;
        }
    }

    (quant, scales, mins)
}
