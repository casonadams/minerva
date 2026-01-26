use super::compute_graph::{ComputeGraph, Node, NodeId, Operation};
use super::graph_fusion::{FusionDetector, FusionPattern};
use std::collections::HashMap;

pub struct GraphOptimizer;

impl GraphOptimizer {
    pub fn optimize(graph: &ComputeGraph) -> ComputeGraph {
        let mut optimized = ComputeGraph::new();
        let mut node_mapping = HashMap::new();
        let mut fused_nodes = std::collections::HashSet::new();
        let mut nodes_to_skip = std::collections::HashSet::new();

        let patterns = FusionDetector::detect_all(graph);
        for (node_id, pattern) in &patterns {
            fused_nodes.insert(*node_id);
            let fusible = FusionDetector::get_fusible_nodes(*pattern, *node_id, graph);
            for fusible_id in fusible {
                if fusible_id != *node_id {
                    nodes_to_skip.insert(fusible_id);
                }
            }
        }

        let order = graph.topological_sort();

        for node_id in order {
            if nodes_to_skip.contains(&node_id) {
                continue;
            }

            if fused_nodes.contains(&node_id) {
                if let Some(node) = graph.get_node(node_id) {
                    let pattern = FusionDetector::detect(node_id, graph);

                    match pattern {
                        FusionPattern::LinearAdd => {
                            Self::add_fused_linear_add(
                                &mut optimized,
                                &mut node_mapping,
                                node_id,
                                graph,
                            );
                        }
                        FusionPattern::LinearGelu => {
                            Self::add_fused_linear_gelu(
                                &mut optimized,
                                &mut node_mapping,
                                node_id,
                                graph,
                            );
                        }
                        FusionPattern::LinearAddGelu => {
                            Self::add_fused_linear_add_gelu(
                                &mut optimized,
                                &mut node_mapping,
                                node_id,
                                graph,
                            );
                        }
                        _ => {
                            Self::copy_node(&mut optimized, &mut node_mapping, node, graph);
                        }
                    }
                }
            } else if let Some(node) = graph.get_node(node_id) {
                Self::copy_node(&mut optimized, &mut node_mapping, node, graph);
            }
        }

        optimized
    }

    fn copy_node(
        optimized: &mut ComputeGraph,
        node_mapping: &mut HashMap<NodeId, NodeId>,
        node: &Node,
        _graph: &ComputeGraph,
    ) {
        let mapped_inputs: Vec<NodeId> = node
            .inputs
            .iter()
            .map(|input_id| node_mapping.get(input_id).copied().unwrap_or(*input_id))
            .collect();

        let new_id = optimized.add_node(node.op.clone(), mapped_inputs);
        node_mapping.insert(node.id, new_id);
    }

    fn add_fused_linear_add(
        optimized: &mut ComputeGraph,
        node_mapping: &mut HashMap<NodeId, NodeId>,
        add_node_id: NodeId,
        graph: &ComputeGraph,
    ) {
        if let Some(add_node) = graph.get_node(add_node_id) {
            if let Some(matmul_node) = graph.get_node(add_node.inputs[0]) {
                if matches!(matmul_node.op, Operation::MatMul { .. }) {
                    if let Operation::MatMul { shape } = matmul_node.op {
                        let mapped_inputs: Vec<NodeId> = add_node
                            .inputs
                            .iter()
                            .flat_map(|input_id| {
                                if let Some(input_node) = graph.get_node(*input_id) {
                                    input_node
                                        .inputs
                                        .iter()
                                        .map(|id| node_mapping.get(id).copied().unwrap_or(*id))
                                        .collect::<Vec<_>>()
                                } else {
                                    vec![node_mapping.get(input_id).copied().unwrap_or(*input_id)]
                                }
                            })
                            .collect();

                        let new_op = Operation::FusedLinearAdd { shape };
                        let new_id = optimized.add_node(new_op, mapped_inputs);
                        node_mapping.insert(add_node_id, new_id);
                    }
                }
            }
        }
    }

    fn add_fused_linear_gelu(
        optimized: &mut ComputeGraph,
        node_mapping: &mut HashMap<NodeId, NodeId>,
        gelu_node_id: NodeId,
        graph: &ComputeGraph,
    ) {
        if let Some(gelu_node) = graph.get_node(gelu_node_id) {
            if let Some(matmul_node) = graph.get_node(gelu_node.inputs[0]) {
                if let Operation::MatMul { shape } = matmul_node.op {
                    let mapped_inputs: Vec<NodeId> = matmul_node
                        .inputs
                        .iter()
                        .map(|input_id| node_mapping.get(input_id).copied().unwrap_or(*input_id))
                        .collect();

                    let new_op = Operation::FusedLinearGelu { shape };
                    let new_id = optimized.add_node(new_op, mapped_inputs);
                    node_mapping.insert(gelu_node_id, new_id);
                }
            }
        }
    }

    fn add_fused_linear_add_gelu(
        optimized: &mut ComputeGraph,
        node_mapping: &mut HashMap<NodeId, NodeId>,
        gelu_node_id: NodeId,
        graph: &ComputeGraph,
    ) {
        if let Some(gelu_node) = graph.get_node(gelu_node_id) {
            if let Some(add_node) = graph.get_node(gelu_node.inputs[0]) {
                if let Some(matmul_node) = graph.get_node(add_node.inputs[0]) {
                    if let Operation::MatMul { shape } = matmul_node.op {
                        let matmul_inputs: Vec<NodeId> = matmul_node
                            .inputs
                            .iter()
                            .map(|id| node_mapping.get(id).copied().unwrap_or(*id))
                            .collect();

                        let add_input: NodeId = add_node
                            .inputs
                            .iter()
                            .find(|id| **id != add_node.inputs[0])
                            .map(|id| node_mapping.get(id).copied().unwrap_or(*id))
                            .unwrap_or(add_node.inputs.get(1).copied().unwrap_or(0));

                        let mut fused_inputs = matmul_inputs;
                        fused_inputs.push(add_input);

                        let new_op = Operation::FusedLinearAddGelu { shape };
                        let new_id = optimized.add_node(new_op, fused_inputs);
                        node_mapping.insert(gelu_node_id, new_id);
                    }
                }
            }
        }
    }
}
