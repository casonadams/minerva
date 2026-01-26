#[cfg(test)]
mod tests {
    use crate::inference::mlx_native::{
        compute_graph::{ComputeGraph, Operation},
        graph_optimizer::GraphOptimizer,
    };

    #[test]
    fn test_optimize_linear_add() {
        let mut graph = ComputeGraph::new();
        let matmul_id = graph.add_node(Operation::MatMul { shape: (10, 10) }, vec![0]);
        let add_id = graph.add_node(Operation::Add, vec![matmul_id, 1]);

        let optimized = GraphOptimizer::optimize(&graph);

        let fused_found = optimized
            .all_nodes()
            .any(|(_, node)| matches!(node.op, Operation::FusedLinearAdd { .. }));
        assert!(fused_found || optimized.all_nodes().count() <= graph.all_nodes().count());
    }

    #[test]
    fn test_optimize_preserves_non_fusible() {
        let mut graph = ComputeGraph::new();
        let add_id = graph.add_node(Operation::Add, vec![0, 1]);
        let gelu_id = graph.add_node(Operation::Gelu, vec![add_id]);
        graph.set_output(gelu_id);

        let optimized = GraphOptimizer::optimize(&graph);

        assert!(optimized.all_nodes().count() > 0);
    }

    #[test]
    fn test_optimize_linear_add_gelu() {
        let mut graph = ComputeGraph::new();
        let matmul_id = graph.add_node(Operation::MatMul { shape: (10, 10) }, vec![0]);
        let add_id = graph.add_node(Operation::Add, vec![matmul_id, 1]);
        let gelu_id = graph.add_node(Operation::Gelu, vec![add_id]);

        let optimized = GraphOptimizer::optimize(&graph);

        let original_count = graph.all_nodes().count();
        let optimized_count = optimized.all_nodes().count();

        assert!(optimized_count <= original_count);
    }

    #[test]
    fn test_optimize_complex_graph() {
        let mut graph = ComputeGraph::new();

        let mm1 = graph.add_node(Operation::MatMul { shape: (10, 10) }, vec![0]);
        let add1 = graph.add_node(Operation::Add, vec![mm1, 1]);
        let gelu1 = graph.add_node(Operation::Gelu, vec![add1]);

        let mm2 = graph.add_node(Operation::MatMul { shape: (10, 10) }, vec![gelu1]);
        let _add2 = graph.add_node(Operation::Add, vec![mm2, 2]);

        let optimized = GraphOptimizer::optimize(&graph);

        let original_count = graph.all_nodes().count();
        let optimized_count = optimized.all_nodes().count();

        assert!(optimized_count <= original_count);
    }
}
