#[cfg(test)]
mod tests {
    use crate::inference::mlx_native::{
        ArrayShape, MLXArray,
        graph_fusion_ops::{
            FusedLinearAddGeluOp, FusedLinearAddOp, FusedLinearGeluOp, FusedOpExecutor,
        },
    };

    #[test]
    fn test_fused_linear_add_correctness() {
        let matmul_lhs = MLXArray::new_cpu(vec![1.0, 2.0, 3.0, 4.0], ArrayShape::Shape2D(2, 2));
        let matmul_rhs = MLXArray::new_cpu(vec![1.0, 0.0, 0.0, 1.0], ArrayShape::Shape2D(2, 2));
        let add_op = MLXArray::new_cpu(vec![10.0, 20.0, 30.0, 40.0], ArrayShape::Shape2D(2, 2));

        let fused = FusedLinearAddOp {
            matmul_shape: (2, 2),
        };
        let result = fused.execute(&[&matmul_lhs, &matmul_rhs, &add_op]);

        let expected = vec![11.0, 22.0, 33.0, 44.0];
        assert_eq!(result.data(), expected);
    }

    #[test]
    fn test_fused_linear_gelu_correctness() {
        let matmul_lhs = MLXArray::new_cpu(vec![1.0, 2.0, 3.0, 4.0], ArrayShape::Shape2D(2, 2));
        let matmul_rhs = MLXArray::new_cpu(vec![1.0, 0.0, 0.0, 1.0], ArrayShape::Shape2D(2, 2));

        let fused = FusedLinearGeluOp {
            matmul_shape: (2, 2),
        };
        let result = fused.execute(&[&matmul_lhs, &matmul_rhs]);

        let data = result.data();
        assert_eq!(data.len(), 4);
        assert!(data[0] > 0.0);
        assert!(data[1] > 0.0);
    }

    #[test]
    fn test_fused_linear_add_gelu_correctness() {
        let matmul_lhs = MLXArray::new_cpu(vec![1.0, 2.0, 3.0, 4.0], ArrayShape::Shape2D(2, 2));
        let matmul_rhs = MLXArray::new_cpu(vec![1.0, 0.0, 0.0, 1.0], ArrayShape::Shape2D(2, 2));
        let add_op = MLXArray::new_cpu(vec![0.0, 0.0, 0.0, 0.0], ArrayShape::Shape2D(2, 2));

        let fused = FusedLinearAddGeluOp {
            matmul_shape: (2, 2),
        };
        let result = fused.execute(&[&matmul_lhs, &matmul_rhs, &add_op]);

        let data = result.data();
        assert_eq!(data.len(), 4);
        assert!(data[0] > 0.0);
        assert!(data[1] > 0.0);
    }

    #[test]
    fn test_fused_linear_add_vs_separate() {
        let matmul_lhs = MLXArray::new_cpu(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ArrayShape::Shape2D(3, 2),
        );
        let matmul_rhs = MLXArray::new_cpu(vec![1.0, 0.0, 0.0, 1.0], ArrayShape::Shape2D(2, 2));
        let add_op = MLXArray::new_cpu(
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            ArrayShape::Shape2D(3, 2),
        );

        let fused = FusedLinearAddOp {
            matmul_shape: (3, 2),
        };
        let fused_result = fused.execute(&[&matmul_lhs, &matmul_rhs, &add_op]);

        let fused_data = fused_result.data();
        assert_eq!(fused_data.len(), 6);
    }

    #[test]
    fn test_fused_ops_memory_efficiency() {
        let matmul_lhs = MLXArray::new_cpu(vec![1.0; 1000], ArrayShape::Shape2D(100, 10));
        let matmul_rhs = MLXArray::new_cpu(vec![1.0; 100], ArrayShape::Shape2D(10, 10));
        let add_op = MLXArray::new_cpu(vec![0.5; 1000], ArrayShape::Shape2D(100, 10));

        let fused = FusedLinearAddGeluOp {
            matmul_shape: (100, 10),
        };
        let result = fused.execute(&[&matmul_lhs, &matmul_rhs, &add_op]);

        assert_eq!(result.data().len(), 1000);
    }
}
