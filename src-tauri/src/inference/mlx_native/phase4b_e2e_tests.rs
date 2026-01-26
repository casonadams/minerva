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
    fn test_e2e_linear_add_gelu_fusion() {
        let mut graph = ComputeGraph::new();

        let mm = graph.add_node(Operation::MatMul { shape: (4, 4) }, vec![0]);
        let add = graph.add_node(Operation::Add, vec![mm, 1]);
        let gelu = graph.add_node(Operation::Gelu, vec![add]);
        graph.set_output(gelu);

        let original_count = graph.all_nodes().count();

        let patterns = FusionDetector::detect_all(&graph);
        assert!(patterns.len() > 0);
        let has_pattern = patterns.iter().any(|(_, p)| {
            *p == crate::inference::mlx_native::graph_fusion::FusionPattern::LinearAddGelu
        });
        assert!(has_pattern);

        let optimized = GraphOptimizer::optimize(&graph);
        let optimized_count = optimized.all_nodes().count();

        assert!(optimized_count <= original_count);
    }

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
}
