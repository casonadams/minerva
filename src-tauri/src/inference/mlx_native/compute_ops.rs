use super::compute_graph::Operation;
use super::unified_memory::MLXArray;
use std::collections::HashMap;

pub trait OpExecutor {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray;
}

pub struct AddExecutor;
impl OpExecutor for AddExecutor {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray {
        assert_eq!(inputs.len(), 2);
        let a = inputs[0].data();
        let b = inputs[1].data();
        let result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        MLXArray::new_cpu(result, inputs[0].shape().clone())
    }
}

pub struct GeluExecutor;
impl OpExecutor for GeluExecutor {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray {
        assert_eq!(inputs.len(), 1);
        let data = inputs[0].data();
        let result: Vec<f32> = data
            .iter()
            .map(|&x| 0.5 * x * (1.0 + (0.7979 * (x + 0.0445 * x * x * x)).tanh()))
            .collect();
        MLXArray::new_cpu(result, inputs[0].shape().clone())
    }
}

pub struct MatMulExecutor {
    pub shape: (usize, usize),
}

impl OpExecutor for MatMulExecutor {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray {
        assert_eq!(inputs.len(), 2);
        let a = inputs[0].data();
        let b = inputs[1].data();
        let (m, n) = self.shape;
        let k = a.len() / m;
        let mut out = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                for p in 0..k {
                    out[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        MLXArray::new_cpu(out, super::unified_memory::ArrayShape::Shape2D(m, n))
    }
}

pub struct LayerNormExecutor {
    pub eps: f32,
}

impl OpExecutor for LayerNormExecutor {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray {
        assert_eq!(inputs.len(), 1);
        let data = inputs[0].data();
        let len = data.len() as f32;
        let mean = data.iter().sum::<f32>() / len;
        let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / len;
        let result: Vec<f32> = data
            .iter()
            .map(|x| (x - mean) / (var + self.eps).sqrt())
            .collect();
        MLXArray::new_cpu(result, inputs[0].shape().clone())
    }
}

pub struct SoftmaxExecutor;
impl OpExecutor for SoftmaxExecutor {
    fn execute(&self, inputs: &[&MLXArray]) -> MLXArray {
        assert_eq!(inputs.len(), 1);
        let data = inputs[0].data();
        if data.is_empty() {
            return MLXArray::new_cpu(Vec::new(), inputs[0].shape().clone());
        }
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = data.iter().map(|x| (x - max).exp()).sum();
        let result: Vec<f32> = data.iter().map(|x| (x - max).exp() / sum).collect();
        MLXArray::new_cpu(result, inputs[0].shape().clone())
    }
}

pub fn execute_op(op: &Operation, inputs: &[&MLXArray]) -> MLXArray {
    match op {
        Operation::Add => AddExecutor.execute(inputs),
        Operation::Gelu => GeluExecutor.execute(inputs),
        Operation::LayerNorm { eps } => LayerNormExecutor { eps: *eps }.execute(inputs),
        Operation::Softmax => SoftmaxExecutor.execute(inputs),
        Operation::MatMul { shape } => MatMulExecutor { shape: *shape }.execute(inputs),
        Operation::Attention { .. } => panic!("Attention not yet implemented"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::mlx_native::ArrayShape;

    #[test]
    fn test_add_executor() {
        let a = MLXArray::new_cpu(vec![1.0, 2.0], ArrayShape::Shape1D(2));
        let b = MLXArray::new_cpu(vec![3.0, 4.0], ArrayShape::Shape1D(2));
        let result = AddExecutor.execute(&[&a, &b]);
        assert_eq!(result.data(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_gelu_executor() {
        let input = MLXArray::new_cpu(vec![0.0], ArrayShape::Shape1D(1));
        let result = GeluExecutor.execute(&[&input]);
        assert!((result.data()[0] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_layernorm_executor() {
        let input = MLXArray::new_cpu(vec![1.0, 2.0, 3.0], ArrayShape::Shape1D(3));
        let result = LayerNormExecutor { eps: 1e-6 }.execute(&[&input]);
        let mean: f32 = result.data().iter().sum::<f32>() / 3.0;
        assert!(mean.abs() < 1e-5);
    }
}
