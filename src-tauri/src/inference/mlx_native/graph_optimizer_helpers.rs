use super::compute_graph::{ComputeGraph, Node, NodeId, Operation};
use std::collections::HashMap;

pub fn copy_node(
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

pub fn add_fused_linear_add(
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

pub fn add_fused_linear_gelu(
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

pub fn add_fused_linear_add_gelu(
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
