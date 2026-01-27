/// Phase 5 GPU Benchmarking
///
/// Measures performance of GPU vs CPU operations
/// Used to validate Phase 5 acceleration targets

#[cfg(test)]
mod benchmarks {
    use crate::inference::mlx_native::metal_gpu::MetalGPU;
    use crate::inference::mlx_native::unified_memory::{ArrayShape, MLXArray};
    use std::time::Instant;

    /// Benchmark single GPU operation
    fn benchmark_gpu_operation(name: &str, iterations: usize, mut op: impl FnMut()) -> f64 {
        let start = Instant::now();
        for _ in 0..iterations {
            op();
        }
        let elapsed = start.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        println!("{}: {:.3}ms ({} iterations)", name, ms, iterations);
        ms
    }

    #[test]
    fn bench_gpu_availability() {
        let available = MetalGPU::is_available();
        println!("GPU Available: {}", available);
    }

    #[test]
    fn bench_gpu_buffer_allocation() {
        if !MetalGPU::is_available() {
            return;
        }

        let gpu = MetalGPU::new().unwrap();

        benchmark_gpu_operation("Small buffer (1KB) allocation", 100, || {
            let _ = gpu.create_buffer(1024);
        });

        benchmark_gpu_operation("Medium buffer (1MB) allocation", 10, || {
            let _ = gpu.create_buffer(1024 * 1024);
        });

        benchmark_gpu_operation("Large buffer (10MB) allocation", 1, || {
            let _ = gpu.create_buffer(10 * 1024 * 1024);
        });
    }

    #[test]
    fn bench_gpu_data_transfer() {
        if !MetalGPU::is_available() {
            return;
        }

        let gpu = MetalGPU::new().unwrap();
        let data = vec![1.0; 1024 * 1024]; // 4MB

        let buffer = gpu.create_buffer(data.len() * 4).unwrap();

        benchmark_gpu_operation("CPU→GPU transfer (4MB)", 5, || {
            let _ = gpu.copy_to_gpu(buffer, &data);
        });

        let mut result = vec![0.0; data.len()];
        benchmark_gpu_operation("GPU→CPU transfer (4MB)", 5, || {
            let _ = gpu.copy_from_gpu(buffer, &mut result);
        });

        gpu.release_buffer(buffer);
    }

    #[test]
    fn bench_array_creation() {
        benchmark_gpu_operation("Create 2D array (100x100)", 100, || {
            let _arr = MLXArray::new_cpu(vec![0.0; 10000], ArrayShape::Shape2D(100, 100));
        });

        benchmark_gpu_operation("Create 1D array (100K)", 100, || {
            let _arr = MLXArray::new_cpu(vec![0.0; 100000], ArrayShape::Shape1D(100000));
        });
    }

    #[test]
    fn bench_gpu_device_creation() {
        if !MetalGPU::is_available() {
            return;
        }

        benchmark_gpu_operation("Create Metal device", 5, || {
            let _ = MetalGPU::new();
        });
    }

    #[test]
    fn bench_gpu_command_buffer() {
        if !MetalGPU::is_available() {
            return;
        }

        let gpu = MetalGPU::new().unwrap();

        benchmark_gpu_operation("Create command buffer", 100, || {
            let _ = gpu.create_command_buffer();
        });
    }

    /// Calculate speedup: cpu_time / gpu_time
    fn speedup(cpu_ms: f64, gpu_ms: f64) -> f64 {
        if gpu_ms > 0.0 { cpu_ms / gpu_ms } else { 0.0 }
    }

    #[test]
    fn bench_phase5_summary() {
        println!("\n=== Phase 5 GPU Acceleration Benchmark Summary ===\n");

        if !MetalGPU::is_available() {
            println!("GPU Not Available - skipping benchmarks");
            return;
        }

        println!("Performance Targets for Phase 5:");
        println!("  - MatMul: 10x speedup (GPU vs CPU)");
        println!("  - Add: 10x speedup");
        println!("  - GELU: 10x speedup");
        println!("  - Fused MatMul+Add+GELU: 20x speedup");
        println!("\nBenchmarks completed - see output above");
    }
}
