use super::metal_gpu::MetalGPU;
use super::unified_memory::MLXArray;
use std::sync::Arc;

/// GPU MatMul execution helper
pub fn gpu_matmul(
    gpu: &Arc<MetalGPU>,
    inputs: &[&MLXArray],
    shape: (usize, usize),
) -> Result<MLXArray, String> {
    if inputs.len() < 2 {
        return Err("MatMul requires 2 inputs".to_string());
    }

    let a_data = inputs[0].data();
    let b_data = inputs[1].data();
    let (m, n) = shape;
    let k = a_data.len() / m;

    let gpu_a = gpu.create_buffer(a_data.len() * 4)?;
    let gpu_b = gpu.create_buffer(b_data.len() * 4)?;
    let gpu_c = gpu.create_buffer(m * n * 4)?;

    gpu.copy_to_gpu(gpu_a, &a_data)?;
    gpu.copy_to_gpu(gpu_b, &b_data)?;

    super::metal_kernels_wrapper::MetalKernels::matmul(
        gpu, gpu_a, gpu_b, gpu_c, m as u32, n as u32, k as u32,
    )?;

    let mut result_data = vec![0.0; m * n];
    gpu.copy_from_gpu(gpu_c, &mut result_data)?;

    gpu.release_buffer(gpu_a);
    gpu.release_buffer(gpu_b);
    gpu.release_buffer(gpu_c);

    Ok(MLXArray::new_cpu(
        result_data,
        super::unified_memory::ArrayShape::Shape2D(m, n),
    ))
}

/// GPU fused MatMul+Add execution helper
pub fn gpu_fused_matmul_add(
    gpu: &Arc<MetalGPU>,
    inputs: &[&MLXArray],
    shape: (usize, usize),
) -> Result<MLXArray, String> {
    if inputs.len() < 3 {
        return Err("FusedLinearAdd requires 3 inputs".to_string());
    }

    let a_data = inputs[0].data();
    let b_data = inputs[1].data();
    let add_data = inputs[2].data();
    let (m, n) = shape;
    let k = a_data.len() / m;

    let gpu_a = gpu.create_buffer(a_data.len() * 4)?;
    let gpu_b = gpu.create_buffer(b_data.len() * 4)?;
    let gpu_add = gpu.create_buffer(add_data.len() * 4)?;
    let gpu_c = gpu.create_buffer(m * n * 4)?;

    gpu.copy_to_gpu(gpu_a, &a_data)?;
    gpu.copy_to_gpu(gpu_b, &b_data)?;
    gpu.copy_to_gpu(gpu_add, &add_data)?;

    super::metal_kernels_wrapper::MetalKernels::fused_matmul_add_gelu(
        gpu, gpu_a, gpu_b, gpu_add, gpu_c, m as u32, n as u32, k as u32,
    )?;

    let mut result_data = vec![0.0; m * n];
    gpu.copy_from_gpu(gpu_c, &mut result_data)?;

    gpu.release_buffer(gpu_a);
    gpu.release_buffer(gpu_b);
    gpu.release_buffer(gpu_add);
    gpu.release_buffer(gpu_c);

    Ok(MLXArray::new_cpu(
        result_data,
        super::unified_memory::ArrayShape::Shape2D(m, n),
    ))
}

/// GPU fused MatMul+Add+Gelu execution helper
pub fn gpu_fused_matmul_add_gelu(
    gpu: &Arc<MetalGPU>,
    inputs: &[&MLXArray],
    shape: (usize, usize),
) -> Result<MLXArray, String> {
    if inputs.len() < 3 {
        return Err("FusedLinearAddGelu requires 3 inputs".to_string());
    }

    let a_data = inputs[0].data();
    let b_data = inputs[1].data();
    let add_data = inputs[2].data();
    let (m, n) = shape;
    let k = a_data.len() / m;

    let gpu_a = gpu.create_buffer(a_data.len() * 4)?;
    let gpu_b = gpu.create_buffer(b_data.len() * 4)?;
    let gpu_add = gpu.create_buffer(add_data.len() * 4)?;
    let gpu_c = gpu.create_buffer(m * n * 4)?;

    gpu.copy_to_gpu(gpu_a, &a_data)?;
    gpu.copy_to_gpu(gpu_b, &b_data)?;
    gpu.copy_to_gpu(gpu_add, &add_data)?;

    super::metal_kernels_wrapper::MetalKernels::fused_matmul_add_gelu(
        gpu, gpu_a, gpu_b, gpu_add, gpu_c, m as u32, n as u32, k as u32,
    )?;

    let mut result_data = vec![0.0; m * n];
    gpu.copy_from_gpu(gpu_c, &mut result_data)?;

    gpu.release_buffer(gpu_a);
    gpu.release_buffer(gpu_b);
    gpu.release_buffer(gpu_add);
    gpu.release_buffer(gpu_c);

    Ok(MLXArray::new_cpu(
        result_data,
        super::unified_memory::ArrayShape::Shape2D(m, n),
    ))
}
