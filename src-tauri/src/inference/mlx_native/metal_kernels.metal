#include <metal_stdlib>
using namespace metal;

// Matrix multiplication kernel (simplified for Apple Silicon)
kernel void matmul_kernel(
    const device float *A,
    const device float *B,
    device float *C,
    constant uint &M,
    constant uint &N,
    constant uint &K,
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}

// Element-wise addition kernel
kernel void add_kernel(
    const device float *A,
    const device float *B,
    device float *C,
    constant uint &size,
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    C[gid] = A[gid] + B[gid];
}

// GELU activation (approximation)
kernel void gelu_kernel(
    const device float *input,
    device float *output,
    constant uint &size,
    uint gid [[thread_position_in_grid]])
{
    if (gid >= size) return;
    
    float x = input[gid];
    float cdf = 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI_F) * (x + 0.044715 * x * x * x)));
    output[gid] = x * cdf;
}

// Fused MatMul + Add + GELU kernel
kernel void fused_matmul_add_gelu_kernel(
    const device float *A,
    const device float *B,
    const device float *bias,
    device float *C,
    constant uint &M,
    constant uint &N,
    constant uint &K,
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    // MatMul
    float sum = 0.0;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    // Add bias
    float x = sum + bias[col];
    
    // Apply GELU
    float cdf = 0.5 * (1.0 + tanh(sqrt(2.0 / M_PI_F) * (x + 0.044715 * x * x * x)));
    C[row * N + col] = x * cdf;
}
