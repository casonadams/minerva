use super::compute_graph::{ComputeGraph, NodeId, Operation};
use super::gpu_execution_helpers::*;
use super::metal_gpu::MetalGPU;
use super::unified_memory::MLXArray;
use std::collections::HashMap;
use std::sync::Arc;

/// GPU-accelerated graph executor
pub struct GPUGraphExecutor {
    gpu: Arc<MetalGPU>,
}

impl GPUGraphExecutor {
    /// Create new GPU executor
    pub fn new() -> Result<Self, String> {
        if !MetalGPU::is_available() {
            return Err("Metal GPU not available".to_string());
        }

        let gpu = Arc::new(MetalGPU::new()?);
        Ok(GPUGraphExecutor { gpu })
    }

    /// Execute graph, routing operations to GPU or CPU
    pub fn execute(
        &self,
        graph: &ComputeGraph,
        inputs: &HashMap<NodeId, MLXArray>,
    ) -> Result<HashMap<NodeId, MLXArray>, String> {
        let mut results = inputs.clone();
        let order = graph.topological_sort();

        for node_id in order {
            if results.contains_key(&node_id) {
                continue;
            }

            if let Some(node) = graph.get_node(node_id) {
                let input_refs: Vec<&MLXArray> = node
                    .inputs
                    .iter()
                    .filter_map(|input_id| results.get(input_id))
                    .collect();

                if input_refs.len() != node.inputs.len() {
                    return Err(format!("Missing inputs for node {}", node_id));
                }

                let output = if self.should_use_gpu(&node.op, &input_refs) {
                    self.execute_on_gpu(&node.op, &input_refs)?
                } else {
                    self.execute_on_cpu(&node.op, &input_refs)?
                };

                results.insert(node_id, output);
            }
        }

        Ok(results)
    }

    /// Decide whether to use GPU for this operation
    fn should_use_gpu(&self, op: &Operation, inputs: &[&MLXArray]) -> bool {
        if inputs.is_empty() {
            return false;
        }

        let data_size: usize = inputs.iter().map(|a| a.data().len()).sum();

        match op {
            Operation::FusedLinearAddGelu { .. } => data_size > 1000,
            Operation::FusedLinearAdd { .. } => data_size > 1000,
            Operation::MatMul { .. } => data_size > 500,
            _ => false,
        }
    }

    /// Execute operation on GPU
    fn execute_on_gpu(&self, op: &Operation, inputs: &[&MLXArray]) -> Result<MLXArray, String> {
        match op {
            Operation::FusedLinearAddGelu { shape } => {
                gpu_fused_matmul_add_gelu(&self.gpu, inputs, *shape)
            }
            Operation::FusedLinearAdd { shape } => gpu_fused_matmul_add(&self.gpu, inputs, *shape),
            Operation::MatMul { shape } => gpu_matmul(&self.gpu, inputs, *shape),
            _ => self.execute_on_cpu(op, inputs),
        }
    }

    /// Execute operation on CPU (fallback)
    fn execute_on_cpu(&self, op: &Operation, inputs: &[&MLXArray]) -> Result<MLXArray, String> {
        use super::compute_ops::execute_op;
        Ok(execute_op(op, inputs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability() {
        // Just check if GPU is available (may be false on non-Apple systems)
        let _available = MetalGPU::is_available();
    }

    #[test]
    fn test_gpu_executor_creation() {
        if !MetalGPU::is_available() {
            return;
        }
        let executor = GPUGraphExecutor::new();
        assert!(executor.is_ok());
    }

    #[test]
    fn test_should_use_gpu() {
        if !MetalGPU::is_available() {
            return;
        }

        let executor = GPUGraphExecutor::new().unwrap();
        let large_array = MLXArray::new_cpu(
            vec![0.0; 2000],
            super::super::unified_memory::ArrayShape::Shape1D(2000),
        );

        let should_use = executor.should_use_gpu(
            &Operation::MatMul { shape: (100, 100) },
            &vec![&large_array, &large_array],
        );

        assert!(should_use, "GPU should be used for large arrays");
    }
}
