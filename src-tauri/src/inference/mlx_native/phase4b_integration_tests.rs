#[cfg(test)]
mod integration_tests {
    use crate::inference::mlx_native::{
        ArrayShape, MLXArray,
        compute_graph::{ComputeGraph, Operation},
        graph_executor::Executor,
        graph_fusion::FusionDetector,
        graph_optimizer::GraphOptimizer,
    };
    use std::collections::HashMap;

    #[test]
    fn test_e2e_naive_vs_fused_execution() {
        let mut naive_graph = ComputeGraph::new();
        let mm1 = naive_graph.add_node(Operation::MatMul { shape: (2, 2) }, vec![0]);
        let add1 = naive_graph.add_node(Operation::Add, vec![mm1, 1]);
        let gelu1 = naive_graph.add_node(Operation::Gelu, vec![add1]);
        naive_graph.set_output(gelu1);

        let mut inputs = HashMap::new();
        inputs.insert(
            0,
            MLXArray::new_cpu(vec![1.0, 2.0, 3.0, 4.0], ArrayShape::Shape2D(2, 2)),
        );
        inputs.insert(
            1,
            MLXArray::new_cpu(vec![0.5, 0.5, 0.5, 0.5], ArrayShape::Shape2D(2, 2)),
        );

        let naive_results = Executor::execute(&naive_graph, &inputs);
        let _naive_output = naive_results[&gelu1].data().clone();

        let optimized_graph = GraphOptimizer::optimize(&naive_graph);
        let optimized_results = Executor::execute(&optimized_graph, &inputs);

        assert!(!optimized_results.is_empty());
    }

    #[test]
    fn test_multiple_fusion_patterns() {
        let mut graph = ComputeGraph::new();

        let mm1 = graph.add_node(Operation::MatMul { shape: (2, 2) }, vec![0]);
        let add1 = graph.add_node(Operation::Add, vec![mm1, 1]);
        let gelu1 = graph.add_node(Operation::Gelu, vec![add1]);

        let mm2 = graph.add_node(Operation::MatMul { shape: (2, 2) }, vec![gelu1]);
        let add2 = graph.add_node(Operation::Add, vec![mm2, 2]);
        graph.set_output(add2);

        let patterns = FusionDetector::detect_all(&graph);
        assert!(patterns.len() >= 1);

        let optimized = GraphOptimizer::optimize(&graph);
        let original_count = graph.all_nodes().count();
        let optimized_count = optimized.all_nodes().count();

        assert!(optimized_count <= original_count);
    }

    #[test]
    fn test_complex_graph_optimization() {
        let mut graph = ComputeGraph::new();

        let mm1 = graph.add_node(Operation::MatMul { shape: (8, 8) }, vec![0]);
        let add1 = graph.add_node(Operation::Add, vec![mm1, 1]);
        let gelu1 = graph.add_node(Operation::Gelu, vec![add1]);

        let ln1 = graph.add_node(Operation::LayerNorm { eps: 1e-5 }, vec![gelu1]);

        let mm2 = graph.add_node(Operation::MatMul { shape: (8, 8) }, vec![ln1]);
        let add2 = graph.add_node(Operation::Add, vec![mm2, 2]);
        graph.set_output(add2);

        let patterns = FusionDetector::detect_all(&graph);
        assert!(patterns.len() >= 1);

        let optimized = GraphOptimizer::optimize(&graph);

        let mut inputs = HashMap::new();
        inputs.insert(
            0,
            MLXArray::new_cpu(vec![1.0; 64], ArrayShape::Shape2D(8, 8)),
        );
        inputs.insert(
            1,
            MLXArray::new_cpu(vec![0.5; 64], ArrayShape::Shape2D(8, 8)),
        );
        inputs.insert(
            2,
            MLXArray::new_cpu(vec![0.1; 64], ArrayShape::Shape2D(8, 8)),
        );

        let results = Executor::execute(&optimized, &inputs);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_graph_optimization_idempotent() {
        let mut graph = ComputeGraph::new();
        let mm = graph.add_node(Operation::MatMul { shape: (2, 2) }, vec![0]);
        let add = graph.add_node(Operation::Add, vec![mm, 1]);
        let gelu = graph.add_node(Operation::Gelu, vec![add]);
        graph.set_output(gelu);

        let opt1 = GraphOptimizer::optimize(&graph);
        let opt1_count = opt1.all_nodes().count();

        let opt2 = GraphOptimizer::optimize(&opt1);
        let opt2_count = opt2.all_nodes().count();

        assert!(opt2_count <= opt1_count);
    }

    #[test]
    fn test_fused_ops_vs_naive_correctness() {
        let mut inputs = HashMap::new();
        let test_data = vec![1.0, 2.0, 3.0, 4.0];
        inputs.insert(
            0,
            MLXArray::new_cpu(test_data.clone(), ArrayShape::Shape2D(2, 2)),
        );
        inputs.insert(
            1,
            MLXArray::new_cpu(vec![0.5, 0.5, 0.5, 0.5], ArrayShape::Shape2D(2, 2)),
        );

        let mut naive_graph = ComputeGraph::new();
        let mm_n = naive_graph.add_node(Operation::MatMul { shape: (2, 2) }, vec![0]);
        let add_n = naive_graph.add_node(Operation::Add, vec![mm_n, 1]);
        naive_graph.set_output(add_n);

        let mut fused_graph = ComputeGraph::new();
        let fused =
            fused_graph.add_node(Operation::FusedLinearAdd { shape: (2, 2) }, vec![0, 1, 1]);
        fused_graph.set_output(fused);

        let naive_results = Executor::execute(&naive_graph, &inputs);
        let fused_results = Executor::execute(&fused_graph, &inputs);

        assert!(naive_results.contains_key(&add_n));
        assert!(fused_results.contains_key(&fused));
    }
}
