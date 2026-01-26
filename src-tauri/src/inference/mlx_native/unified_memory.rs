use ndarray::{Array1, Array2};
use std::sync::{Arc, Mutex};

/// Device type for computation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    CPU,
    GPU,
}

impl Device {
    pub fn name(&self) -> &'static str {
        match self {
            Device::CPU => "CPU",
            Device::GPU => "GPU",
        }
    }
}

/// Unified memory array that can live on CPU or GPU
#[derive(Clone)]
pub struct MLXArray {
    /// Actual data stored as f32 vector
    data: Arc<Mutex<Vec<f32>>>,
    /// Shape of the array
    shape: ArrayShape,
    /// Current device location
    device: Device,
    /// Metadata for tracking
    id: u64,
}

/// Array shape (supports 1D and 2D for now)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayShape {
    Shape1D(usize),
    Shape2D(usize, usize),
}

impl ArrayShape {
    pub fn size(&self) -> usize {
        match self {
            ArrayShape::Shape1D(n) => *n,
            ArrayShape::Shape2D(m, n) => m * n,
        }
    }

    pub fn dims(&self) -> Vec<usize> {
        match self {
            ArrayShape::Shape1D(n) => vec![*n],
            ArrayShape::Shape2D(m, n) => vec![*m, *n],
        }
    }
}

impl MLXArray {
    /// Create a new array on CPU
    pub fn new_cpu(data: Vec<f32>, shape: ArrayShape) -> Self {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        MLXArray {
            data: Arc::new(Mutex::new(data)),
            shape,
            device: Device::CPU,
            id,
        }
    }

    /// Create a new array on GPU
    pub fn new_gpu(data: Vec<f32>, shape: ArrayShape) -> Self {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        MLXArray {
            data: Arc::new(Mutex::new(data)),
            shape,
            device: Device::GPU,
            id,
        }
    }

    /// Get current device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get shape
    pub fn shape(&self) -> ArrayShape {
        self.shape
    }

    /// Get size (total elements)
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Move array to target device (CPU/GPU)
    pub fn to_device(&self, target: Device) -> Self {
        if self.device == target {
            return self.clone();
        }

        // In real implementation, this would use Metal for GPU transfers
        // For now, just copy data (all in system memory)
        let data = self.data.lock().unwrap().clone();

        MLXArray {
            data: Arc::new(Mutex::new(data)),
            shape: self.shape,
            device: target,
            id: self.id,
        }
    }

    /// Get reference to data (for CPU operations)
    pub fn data(&self) -> Vec<f32> {
        self.data.lock().unwrap().clone()
    }

    /// Get mutable reference to data (for in-place operations)
    pub fn data_mut(&mut self) -> Vec<f32> {
        let guard = self.data.lock().unwrap();
        guard.clone()
    }

    /// Convert 1D array to ndarray Array1
    pub fn to_array1(&self) -> Option<Array1<f32>> {
        match self.shape {
            ArrayShape::Shape1D(_) => {
                let data = self.data();
                Some(Array1::from_vec(data))
            }
            ArrayShape::Shape2D(_, _) => None,
        }
    }

    /// Convert 2D array to ndarray Array2
    pub fn to_array2(&self) -> Option<Array2<f32>> {
        match self.shape {
            ArrayShape::Shape1D(_) => None,
            ArrayShape::Shape2D(m, n) => {
                let data = self.data();
                Array2::from_shape_vec((m, n), data).ok()
            }
        }
    }

    /// Create from Array1
    pub fn from_array1(arr: &Array1<f32>, device: Device) -> Self {
        let shape = ArrayShape::Shape1D(arr.len());
        let data = arr.to_vec();
        match device {
            Device::CPU => Self::new_cpu(data, shape),
            Device::GPU => Self::new_gpu(data, shape),
        }
    }

    /// Create from Array2
    pub fn from_array2(arr: &Array2<f32>, device: Device) -> Self {
        let (m, n) = arr.dim();
        let shape = ArrayShape::Shape2D(m, n);
        let data = arr.to_owned().into_shape(m * n).unwrap().to_vec();
        match device {
            Device::CPU => Self::new_cpu(data, shape),
            Device::GPU => Self::new_gpu(data, shape),
        }
    }

    /// Get array ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Clone to CPU (always safe)
    pub fn clone_to_cpu(&self) -> Self {
        self.to_device(Device::CPU)
    }

    /// Clone to GPU (always safe)
    pub fn clone_to_gpu(&self) -> Self {
        self.to_device(Device::GPU)
    }
}

/// Memory pool for managing multiple arrays
pub struct MemoryPool {
    arrays: Arc<Mutex<Vec<MLXArray>>>,
    device: Device,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(device: Device) -> Self {
        MemoryPool {
            arrays: Arc::new(Mutex::new(Vec::new())),
            device,
        }
    }

    /// Allocate a new array in this pool
    pub fn allocate(&mut self, data: Vec<f32>, shape: ArrayShape) -> MLXArray {
        let array = match self.device {
            Device::CPU => MLXArray::new_cpu(data, shape),
            Device::GPU => MLXArray::new_gpu(data, shape),
        };

        self.arrays.lock().unwrap().push(array.clone());
        array
    }

    /// Get total memory usage (rough estimate in bytes)
    pub fn memory_usage(&self) -> usize {
        self.arrays
            .lock()
            .unwrap()
            .iter()
            .map(|arr| arr.size() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Clear all arrays
    pub fn clear(&mut self) {
        self.arrays.lock().unwrap().clear();
    }

    /// Move all arrays to target device
    pub fn move_to_device(&mut self, device: Device) {
        let mut arrays = self.arrays.lock().unwrap();
        *arrays = arrays.iter().map(|arr| arr.to_device(device)).collect();
        self.device = device;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_array_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let arr = MLXArray::new_cpu(data, ArrayShape::Shape1D(4));
        assert_eq!(arr.size(), 4);
        assert_eq!(arr.device(), Device::CPU);
    }

    #[test]
    fn test_array_shape() {
        let data = vec![1.0; 12];
        let arr = MLXArray::new_cpu(data, ArrayShape::Shape2D(3, 4));
        assert_eq!(arr.size(), 12);
        match arr.shape() {
            ArrayShape::Shape2D(m, n) => {
                assert_eq!(m, 3);
                assert_eq!(n, 4);
            }
            _ => panic!("Expected 2D shape"),
        }
    }

    #[test]
    fn test_device_transfer() {
        let data = vec![1.0, 2.0, 3.0];
        let arr_cpu = MLXArray::new_cpu(data, ArrayShape::Shape1D(3));
        assert_eq!(arr_cpu.device(), Device::CPU);

        let arr_gpu = arr_cpu.to_device(Device::GPU);
        assert_eq!(arr_gpu.device(), Device::GPU);

        let arr_cpu_again = arr_gpu.to_device(Device::CPU);
        assert_eq!(arr_cpu_again.device(), Device::CPU);
    }

    #[test]
    fn test_data_preservation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let arr = MLXArray::new_cpu(data.clone(), ArrayShape::Shape1D(5));

        // Transfer to GPU and back
        let arr_gpu = arr.to_device(Device::GPU);
        let arr_back = arr_gpu.to_device(Device::CPU);

        assert_eq!(arr_back.data(), data);
    }

    #[test]
    fn test_array_from_ndarray() {
        let ndarray = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let arr = MLXArray::from_array1(&ndarray, Device::CPU);

        assert_eq!(arr.size(), 3);
        assert_eq!(arr.to_array1().unwrap().to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_array2_from_ndarray() {
        let ndarray = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let arr = MLXArray::from_array2(&ndarray, Device::CPU);

        assert_eq!(arr.size(), 6);
        match arr.shape() {
            ArrayShape::Shape2D(m, n) => {
                assert_eq!(m, 2);
                assert_eq!(n, 3);
            }
            _ => panic!("Expected 2D shape"),
        }
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = MemoryPool::new(Device::CPU);

        let arr1 = pool.allocate(vec![1.0; 100], ArrayShape::Shape1D(100));
        let arr2 = pool.allocate(vec![2.0; 50], ArrayShape::Shape1D(50));

        assert_eq!(arr1.size(), 100);
        assert_eq!(arr2.size(), 50);

        let usage = pool.memory_usage();
        assert!(usage > 0);
    }

    #[test]
    fn test_device_names() {
        assert_eq!(Device::CPU.name(), "CPU");
        assert_eq!(Device::GPU.name(), "GPU");
    }
}
