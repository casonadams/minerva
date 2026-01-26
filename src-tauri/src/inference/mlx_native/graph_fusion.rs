use super::compute_graph::{ComputeGraph, NodeId, Operation};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionPattern {
    None,
    LinearAdd,     // MatMul + Add
    LinearGelu,    // MatMul + Gelu
    LinearAddGelu, // MatMul + Add + Gelu
    AttentionAdd,  // Attention + Add
    SoftmaxScale,  // Softmax + Scale
}

pub struct FusionDetector;

impl FusionDetector {
    pub fn detect_all(graph: &ComputeGraph) -> Vec<(NodeId, FusionPattern)> {
        let mut patterns = Vec::new();

        for (node_id, _node) in graph.all_nodes() {
            let pattern = Self::detect(*node_id, graph);
            if pattern != FusionPattern::None {
                patterns.push((*node_id, pattern));
            }
        }

        patterns
    }

    pub fn detect(node_id: NodeId, graph: &ComputeGraph) -> FusionPattern {
        if let Some(node) = graph.get_node(node_id) {
            match &node.op {
                Operation::Add => Self::detect_add_pattern(node_id, graph),
                Operation::Gelu => Self::detect_gelu_pattern(node_id, graph),
                _ => FusionPattern::None,
            }
        } else {
            FusionPattern::None
        }
    }

    fn detect_add_pattern(node_id: NodeId, graph: &ComputeGraph) -> FusionPattern {
        if let Some(node) = graph.get_node(node_id) {
            if node.inputs.len() != 2 {
                return FusionPattern::None;
            }

            let input0 = graph.get_node(node.inputs[0]);
            let input1 = graph.get_node(node.inputs[1]);

            match (input0, input1) {
                (Some(n0), Some(n1)) => {
                    // Check for MatMul + Add pattern
                    if matches!(n0.op, Operation::MatMul { .. }) {
                        // Check if second add input could be residual
                        return FusionPattern::LinearAdd;
                    }
                    if matches!(n1.op, Operation::MatMul { .. }) {
                        return FusionPattern::LinearAdd;
                    }
                    FusionPattern::None
                }
                _ => FusionPattern::None,
            }
        } else {
            FusionPattern::None
        }
    }

    fn detect_gelu_pattern(node_id: NodeId, graph: &ComputeGraph) -> FusionPattern {
        if let Some(node) = graph.get_node(node_id) {
            if node.inputs.is_empty() {
                return FusionPattern::None;
            }

            if let Some(input_node) = graph.get_node(node.inputs[0]) {
                match &input_node.op {
                    Operation::MatMul { .. } => FusionPattern::LinearGelu,
                    Operation::Add => {
                        // Check if Add has MatMul input (MatMul + Add + Gelu)
                        if input_node.inputs.is_empty() {
                            return FusionPattern::None;
                        }

                        if let Some(add_input) = graph.get_node(input_node.inputs[0]) {
                            if matches!(add_input.op, Operation::MatMul { .. }) {
                                return FusionPattern::LinearAddGelu;
                            }
                        }

                        if let Some(add_input) =
                            graph.get_node(input_node.inputs.get(1).copied().unwrap_or(0))
                        {
                            if matches!(add_input.op, Operation::MatMul { .. }) {
                                return FusionPattern::LinearAddGelu;
                            }
                        }

                        FusionPattern::None
                    }
                    _ => FusionPattern::None,
                }
            } else {
                FusionPattern::None
            }
        } else {
            FusionPattern::None
        }
    }

    pub fn get_fusible_nodes(
        pattern: FusionPattern,
        node_id: NodeId,
        graph: &ComputeGraph,
    ) -> Vec<NodeId> {
        let mut nodes = vec![node_id];

        if let Some(node) = graph.get_node(node_id) {
            match pattern {
                FusionPattern::LinearAdd => {
                    nodes.extend(&node.inputs);
                }
                FusionPattern::LinearGelu => {
                    nodes.extend(&node.inputs);
                }
                FusionPattern::LinearAddGelu => {
                    // Collect all 3 nodes: MatMul + Add + Gelu
                    nodes.extend(&node.inputs);
                    if let Some(add_node) = graph.get_node(node.inputs[0]) {
                        nodes.extend(&add_node.inputs);
                    }
                }
                _ => {}
            }
        }

        // Deduplicate
        let mut unique: Vec<NodeId> = nodes
            .into_iter()
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        unique.sort();
        unique
    }
}
