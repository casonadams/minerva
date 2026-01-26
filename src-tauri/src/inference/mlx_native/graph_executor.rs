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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::mlx_native::{ArrayShape, compute_graph::Operation};

    #[test]
    fn test_execute_single_op() {
        let mut graph = ComputeGraph::new();
        // Reference external input nodes 100, 101 (not in graph)
        let add_id = graph.add_node(Operation::Add, vec![100, 101]);
        graph.set_output(add_id);

        let mut inputs = HashMap::new();
        inputs.insert(
            100,
            MLXArray::new_cpu(vec![1.0, 2.0], ArrayShape::Shape1D(2)),
        );
        inputs.insert(
            101,
            MLXArray::new_cpu(vec![3.0, 4.0], ArrayShape::Shape1D(2)),
        );

        let results = Executor::execute(&graph, &inputs);
        assert!(results.contains_key(&add_id));
        assert_eq!(results[&add_id].data(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_execute_chain() {
        let mut graph = ComputeGraph::new();
        let n0 = graph.add_node(Operation::Add, vec![0, 1]);
        let n1 = graph.add_node(Operation::Gelu, vec![n0]);
        graph.set_output(n1);

        let mut inputs = HashMap::new();
        inputs.insert(0, MLXArray::new_cpu(vec![0.5], ArrayShape::Shape1D(1)));
        inputs.insert(1, MLXArray::new_cpu(vec![0.5], ArrayShape::Shape1D(1)));

        let results = Executor::execute(&graph, &inputs);
        assert!(results.contains_key(&n1));
        assert!(results[&n1].data()[0] > 0.0);
    }

    #[test]
    fn test_execute_preserves_inputs() {
        let mut graph = ComputeGraph::new();
        let n0 = graph.add_node(Operation::Gelu, vec![0]);
        graph.set_output(n0);

        let mut inputs = HashMap::new();
        let data = vec![1.0, 2.0];
        inputs.insert(0, MLXArray::new_cpu(data.clone(), ArrayShape::Shape1D(2)));

        let results = Executor::execute(&graph, &inputs);
        assert_eq!(results[&0].data(), data);
    }
}
