use super::compute_graph::{ComputeGraph, NodeId};
use super::compute_ops::execute_op;
use super::unified_memory::MLXArray;
use std::collections::HashMap;

pub struct Executor;

impl Executor {
    pub fn execute(
        graph: &ComputeGraph,
        inputs: &HashMap<NodeId, MLXArray>,
    ) -> HashMap<NodeId, MLXArray> {
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

                assert_eq!(
                    input_refs.len(),
                    node.inputs.len(),
                    "Missing inputs for node {}",
                    node_id
                );

                let output = execute_op(&node.op, &input_refs);
                results.insert(node_id, output);
            }
        }

        results
    }
}
