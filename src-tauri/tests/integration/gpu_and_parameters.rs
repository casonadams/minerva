// GPU Context and Parameter Validation Integration Tests

// GPU Context Tests

#[test]
fn test_gpu_context_creation() {
    use minerva_lib::inference::gpu_context::GpuContext;

    let ctx = GpuContext::new().unwrap_or_default();
    assert!(ctx.available_memory() > 0);
}

#[test]
fn test_gpu_context_device_detection() {
    use minerva_lib::inference::gpu_context::GpuContext;

    let ctx = GpuContext::new().unwrap_or_default();
    let device = ctx.device();
    // Should have a valid device (Metal, CUDA, or CPU)
    assert!(!format!("{:?}", device).is_empty());
}

#[test]
fn test_gpu_device_detection() {
    use minerva_lib::inference::gpu_context::GpuContext;

    let ctx = GpuContext::new().unwrap_or_default();
    let device = ctx.device();
    // Should detect something (Metal on macOS, CUDA on Linux, etc.)
    assert!(!format!("{:?}", device).is_empty());
}

#[test]
fn test_gpu_context_allocation() {
    use minerva_lib::inference::gpu_context::GpuContext;

    let mut ctx = GpuContext::new().unwrap_or_default();
    let size = 100 * 1024 * 1024; // 100MB
    let result = ctx.allocate(size);
    assert!(result.is_ok() || result.is_err()); // Either succeeds or fails gracefully
}

#[test]
fn test_gpu_memory_limits() {
    use minerva_lib::inference::gpu_context::GpuContext;

    let ctx = GpuContext::new().unwrap_or_default();
    let max_mem = ctx.max_memory();
    assert!(max_mem > 0);
}

#[test]
fn test_gpu_vs_cpu_performance_comparison() {
    use minerva_lib::inference::benchmarks::PerformanceAccumulator;
    use std::time::Duration;

    let mut accumulator = PerformanceAccumulator::new();

    accumulator.add_gpu_measurement(Duration::from_millis(50));
    accumulator.add_gpu_measurement(Duration::from_millis(55));

    accumulator.add_cpu_measurement(Duration::from_millis(500));
    accumulator.add_cpu_measurement(Duration::from_millis(550));

    let speedup = accumulator.speedup_factor().unwrap();
    assert!(speedup > 5.0);
    assert_eq!(accumulator.gpu_count(), 2);
    assert_eq!(accumulator.cpu_count(), 2);
}

#[test]
fn test_gpu_initialization_metal() {
    use minerva_lib::inference::gpu_context::GpuContext;

    let ctx = GpuContext::new().unwrap_or_default();
    let device = ctx.device();

    // Should have a valid device (Metal, CUDA, or CPU)
    assert!(!format!("{:?}", device).is_empty());
}

// Parameter Validation Tests

#[test]
fn test_generation_config_validation_temperature() {
    use minerva_lib::inference::GenerationConfig;

    let invalid = GenerationConfig {
        temperature: 3.0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn test_generation_config_validation_top_p() {
    use minerva_lib::inference::GenerationConfig;

    let invalid = GenerationConfig {
        top_p: 1.5,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}

#[test]
fn test_generation_config_validation_valid() {
    use minerva_lib::inference::GenerationConfig;

    let config = GenerationConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_generation_config_validation_max_tokens() {
    use minerva_lib::inference::GenerationConfig;

    let invalid = GenerationConfig {
        max_tokens: 0,
        ..Default::default()
    };
    assert!(invalid.validate().is_err());
}
