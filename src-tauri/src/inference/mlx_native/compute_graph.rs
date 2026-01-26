use std::collections::HashMap;

pub type NodeId = usize;

#[derive(Clone, Debug)]
pub enum Operation {
    MatMul { shape: (usize, usize) },
    Add,
    Gelu,
    LayerNorm { eps: f32 },
    Softmax,
    Attention { scale: f32 },
    FusedLinearAdd { shape: (usize, usize) },
    FusedLinearGelu { shape: (usize, usize) },
    FusedLinearAddGelu { shape: (usize, usize) },
}

#[derive(Clone)]
pub struct Node {
    pub id: NodeId,
    pub op: Operation,
    pub inputs: Vec<NodeId>,
}

pub struct ComputeGraph {
    nodes: HashMap<NodeId, Node>,
    next_id: NodeId,
    outputs: Vec<NodeId>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        ComputeGraph {
            nodes: HashMap::new(),
            next_id: 0,
            outputs: Vec::new(),
        }
    }

    pub fn add_node(&mut self, op: Operation, inputs: Vec<NodeId>) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, Node { id, op, inputs });
        id
    }

    pub fn set_output(&mut self, node_id: NodeId) {
        if !self.outputs.contains(&node_id) {
            self.outputs.push(node_id);
        }
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn all_nodes(&self) -> impl Iterator<Item = (&NodeId, &Node)> {
        self.nodes.iter()
    }

    pub fn topological_sort(&self) -> Vec<NodeId> {
        let mut visited = std::collections::HashSet::new();
        let mut order = Vec::new();

        for &node_id in &self.outputs {
            self.visit(node_id, &mut visited, &mut order);
        }

        order
    }

    fn visit(
        &self,
        node_id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) {
        if visited.contains(&node_id) {
            return;
        }

        visited.insert(node_id);

        if let Some(node) = self.nodes.get(&node_id) {
            for &input_id in &node.inputs {
                self.visit(input_id, visited, order);
            }
        }

        order.push(node_id);
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_graph() {
        let graph = ComputeGraph::new();
        assert_eq!(graph.nodes.len(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut graph = ComputeGraph::new();
        let id = graph.add_node(Operation::Add, vec![]);
        assert_eq!(id, 0);
        assert_eq!(graph.nodes.len(), 1);
    }

    #[test]
    fn test_topological_sort() {
        let mut graph = ComputeGraph::new();
        let n0 = graph.add_node(Operation::Add, vec![]);
        let n1 = graph.add_node(Operation::Gelu, vec![n0]);
        let n2 = graph.add_node(Operation::Add, vec![n1]);
        graph.set_output(n2);

        let order = graph.topological_sort();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_topological_sort_diamond() {
        let mut graph = ComputeGraph::new();
        let n0 = graph.add_node(Operation::Add, vec![]);
        let n1 = graph.add_node(Operation::Gelu, vec![n0]);
        let n2 = graph.add_node(Operation::Gelu, vec![n0]);
        let n3 = graph.add_node(Operation::Add, vec![n1, n2]);
        graph.set_output(n3);

        let order = graph.topological_sort();
        assert!(
            order.iter().position(|&id| id == 0).unwrap()
                < order.iter().position(|&id| id == 3).unwrap()
        );
    }
}
