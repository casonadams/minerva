use super::unified_memory::MLXArray;

pub trait FusedOpExecutor {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray;
}

/// Fuses MatMul + Add into single operation
/// Input order: [matmul_lhs, matmul_rhs, add_operand]
pub struct FusedLinearAddOp {
    pub matmul_shape: (usize, usize),
}

impl FusedOpExecutor for FusedLinearAddOp {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray {
        assert_eq!(inputs.len(), 3, "FusedLinearAdd expects 3 inputs");

        let matmul_lhs = inputs[0].data();
        let matmul_rhs = inputs[1].data();
        let add_operand = inputs[2].data();

        let (m, n) = self.matmul_shape;
        let k = matmul_lhs.len() / m;

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += matmul_lhs[i * k + p] * matmul_rhs[p * n + j];
                }
                result[i * n + j] = sum + add_operand[i * n + j];
            }
        }

        MLXArray::new_cpu(result, super::unified_memory::ArrayShape::Shape2D(m, n))
    }
}

/// Fuses MatMul + Gelu into single operation
/// Input order: [matmul_lhs, matmul_rhs]
pub struct FusedLinearGeluOp {
    pub matmul_shape: (usize, usize),
}

impl FusedOpExecutor for FusedLinearGeluOp {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray {
        assert_eq!(inputs.len(), 2, "FusedLinearGelu expects 2 inputs");

        let matmul_lhs = inputs[0].data();
        let matmul_rhs = inputs[1].data();

        let (m, n) = self.matmul_shape;
        let k = matmul_lhs.len() / m;

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += matmul_lhs[i * k + p] * matmul_rhs[p * n + j];
                }
                let x = sum;
                result[i * n + j] = 0.5 * x * (1.0 + (0.7979 * (x + 0.0445 * x * x * x)).tanh());
            }
        }

        MLXArray::new_cpu(result, super::unified_memory::ArrayShape::Shape2D(m, n))
    }
}

/// Fuses MatMul + Add + Gelu into single operation (most common in transformers)
/// Input order: [matmul_lhs, matmul_rhs, add_operand]
pub struct FusedLinearAddGeluOp {
    pub matmul_shape: (usize, usize),
}

impl FusedOpExecutor for FusedLinearAddGeluOp {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray {
        assert_eq!(inputs.len(), 3, "FusedLinearAddGelu expects 3 inputs");

        let matmul_lhs = inputs[0].data();
        let matmul_rhs = inputs[1].data();
        let add_operand = inputs[2].data();

        let (m, n) = self.matmul_shape;
        let k = matmul_lhs.len() / m;

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += matmul_lhs[i * k + p] * matmul_rhs[p * n + j];
                }
                let x = sum + add_operand[i * n + j];
                result[i * n + j] = 0.5 * x * (1.0 + (0.7979 * (x + 0.0445 * x * x * x)).tanh());
            }
        }

        MLXArray::new_cpu(result, super::unified_memory::ArrayShape::Shape2D(m, n))
    }
}
