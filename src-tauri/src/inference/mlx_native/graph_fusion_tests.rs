#[cfg(test)]
mod tests {
    use crate::inference::mlx_native::{
        compute_graph::{ComputeGraph, Operation},
        graph_fusion::{FusionDetector, FusionPattern},
    };

    #[test]
    fn test_detect_linear_add() {
        let mut graph = ComputeGraph::new();
        let matmul_id = graph.add_node(Operation::MatMul { shape: (10, 10) }, vec![0]);
        let add_id = graph.add_node(Operation::Add, vec![matmul_id, 1]);

        let pattern = FusionDetector::detect(add_id, &graph);
        assert_eq!(pattern, FusionPattern::LinearAdd);
    }

    #[test]
    fn test_detect_linear_gelu() {
        let mut graph = ComputeGraph::new();
        let matmul_id = graph.add_node(Operation::MatMul { shape: (10, 10) }, vec![0]);
        let gelu_id = graph.add_node(Operation::Gelu, vec![matmul_id]);

        let pattern = FusionDetector::detect(gelu_id, &graph);
        assert_eq!(pattern, FusionPattern::LinearGelu);
    }

    #[test]
    fn test_detect_linear_add_gelu() {
        let mut graph = ComputeGraph::new();
        let matmul_id = graph.add_node(Operation::MatMul { shape: (10, 10) }, vec![0]);
        let add_id = graph.add_node(Operation::Add, vec![matmul_id, 1]);
        let gelu_id = graph.add_node(Operation::Gelu, vec![add_id]);

        let pattern = FusionDetector::detect(gelu_id, &graph);
        assert_eq!(pattern, FusionPattern::LinearAddGelu);
    }

    #[test]
    fn test_detect_all() {
        let mut graph = ComputeGraph::new();
        let n0 = graph.add_node(Operation::MatMul { shape: (10, 10) }, vec![0]);
        let n1 = graph.add_node(Operation::Add, vec![n0, 1]);
        let _n2 = graph.add_node(Operation::Gelu, vec![n1]);

        let patterns = FusionDetector::detect_all(&graph);
        assert!(patterns.len() >= 1);
        assert!(
            patterns
                .iter()
                .any(|(_, p)| p == &FusionPattern::LinearAdd || p == &FusionPattern::LinearAddGelu)
        );
    }

    #[test]
    fn test_get_fusible_nodes() {
        let mut graph = ComputeGraph::new();
        let n0 = graph.add_node(Operation::MatMul { shape: (10, 10) }, vec![0]);
        let n1 = graph.add_node(Operation::Add, vec![n0, 1]);

        let nodes = FusionDetector::get_fusible_nodes(FusionPattern::LinearAdd, n1, &graph);
        assert!(nodes.contains(&n0));
        assert!(nodes.contains(&n1));
    }
}
