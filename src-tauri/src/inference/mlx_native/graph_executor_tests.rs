#[cfg(test)]
mod tests {
    use crate::inference::mlx_native::{
        ArrayShape, MLXArray,
        compute_graph::{ComputeGraph, Operation},
        graph_executor::Executor,
    };
    use std::collections::HashMap;

    #[test]
    fn test_execute_single_op() {
        let mut graph = ComputeGraph::new();
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

    #[test]
    fn test_execute_fused_linear_add() {
        let mut graph = ComputeGraph::new();
        let fused_id = graph.add_node(Operation::FusedLinearAdd { shape: (2, 2) }, vec![0, 1, 2]);
        graph.set_output(fused_id);

        let mut inputs = HashMap::new();
        inputs.insert(
            0,
            MLXArray::new_cpu(vec![1.0, 2.0, 3.0, 4.0], ArrayShape::Shape2D(2, 2)),
        );
        inputs.insert(
            1,
            MLXArray::new_cpu(vec![1.0, 0.0, 0.0, 1.0], ArrayShape::Shape2D(2, 2)),
        );
        inputs.insert(
            2,
            MLXArray::new_cpu(vec![10.0, 20.0, 30.0, 40.0], ArrayShape::Shape2D(2, 2)),
        );

        let results = Executor::execute(&graph, &inputs);
        assert!(results.contains_key(&fused_id));
        let result_data = results[&fused_id].data();
        assert_eq!(result_data.len(), 4);
        for &val in result_data.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_execute_fused_linear_gelu() {
        let mut graph = ComputeGraph::new();
        let fused_id = graph.add_node(Operation::FusedLinearGelu { shape: (2, 2) }, vec![0, 1]);
        graph.set_output(fused_id);

        let mut inputs = HashMap::new();
        inputs.insert(
            0,
            MLXArray::new_cpu(vec![1.0, 2.0, 3.0, 4.0], ArrayShape::Shape2D(2, 2)),
        );
        inputs.insert(
            1,
            MLXArray::new_cpu(vec![1.0, 0.0, 0.0, 1.0], ArrayShape::Shape2D(2, 2)),
        );

        let results = Executor::execute(&graph, &inputs);
        assert!(results.contains_key(&fused_id));
        let result_data = results[&fused_id].data();
        assert_eq!(result_data.len(), 4);
        assert!(result_data[0] > 0.0);
    }

    #[test]
    fn test_execute_fused_linear_add_gelu() {
        let mut graph = ComputeGraph::new();
        let fused_id = graph.add_node(
            Operation::FusedLinearAddGelu { shape: (2, 2) },
            vec![0, 1, 2],
        );
        graph.set_output(fused_id);

        let mut inputs = HashMap::new();
        inputs.insert(
            0,
            MLXArray::new_cpu(vec![1.0, 2.0, 3.0, 4.0], ArrayShape::Shape2D(2, 2)),
        );
        inputs.insert(
            1,
            MLXArray::new_cpu(vec![1.0, 0.0, 0.0, 1.0], ArrayShape::Shape2D(2, 2)),
        );
        inputs.insert(
            2,
            MLXArray::new_cpu(vec![0.0, 0.0, 0.0, 0.0], ArrayShape::Shape2D(2, 2)),
        );

        let results = Executor::execute(&graph, &inputs);
        assert!(results.contains_key(&fused_id));
        let result_data = results[&fused_id].data();
        assert_eq!(result_data.len(), 4);
        assert!(result_data[0] > 0.0);
    }
}
