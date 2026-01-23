use parking_lot::RwLock;
/// GPU Batch Scheduling for Phase 5
///
/// This module provides GPU-accelerated batch processing using Metal framework on macOS.
/// It includes GPU memory management, batch scheduling, and compute shader integration.
use std::collections::VecDeque;
use std::sync::Arc;

/// GPU memory allocation tracker
#[derive(Clone, Debug)]
pub struct GPUMemoryAllocation {
    pub buffer_id: u32,
    pub size_bytes: usize,
    pub offset_bytes: usize,
}

impl GPUMemoryAllocation {
    pub fn new(buffer_id: u32, size_bytes: usize, offset_bytes: usize) -> Self {
        Self {
            buffer_id,
            size_bytes,
            offset_bytes,
        }
    }
}

/// GPU memory pool for batch operations
pub struct GPUMemoryPool {
    total_memory: usize,
    allocated_memory: usize,
    free_blocks: VecDeque<GPUMemoryAllocation>,
    allocated_blocks: Vec<GPUMemoryAllocation>,
}

impl GPUMemoryPool {
    pub fn new(total_memory: usize) -> Self {
        let initial_block = GPUMemoryAllocation::new(0, total_memory, 0);
        let mut free_blocks = VecDeque::new();
        free_blocks.push_back(initial_block);

        Self {
            total_memory,
            allocated_memory: 0,
            free_blocks,
            allocated_blocks: Vec::new(),
        }
    }

    /// Allocate memory from the pool
    pub fn allocate(&mut self, size_bytes: usize) -> Option<GPUMemoryAllocation> {
        if size_bytes > self.total_memory - self.allocated_memory {
            return None; // Not enough memory
        }

        // Find a suitable free block (first-fit)
        for block in &self.free_blocks {
            if block.size_bytes >= size_bytes {
                let allocation =
                    GPUMemoryAllocation::new(block.buffer_id, size_bytes, block.offset_bytes);
                self.allocated_blocks.push(allocation.clone());
                self.allocated_memory += size_bytes;
                return Some(allocation);
            }
        }

        None
    }

    /// Deallocate memory back to the pool
    pub fn deallocate(&mut self, allocation: &GPUMemoryAllocation) {
        self.allocated_memory = self.allocated_memory.saturating_sub(allocation.size_bytes);
        self.allocated_blocks
            .retain(|b| b.buffer_id != allocation.buffer_id);
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> (usize, usize) {
        (self.allocated_memory, self.total_memory)
    }

    /// Get memory usage percentage
    pub fn memory_usage_percent(&self) -> f32 {
        if self.total_memory == 0 {
            0.0
        } else {
            (self.allocated_memory as f32 / self.total_memory as f32) * 100.0
        }
    }
}

/// GPU batch item for scheduling
#[derive(Clone, Debug)]
pub struct GPUBatchItem<T: Clone> {
    pub id: String,
    pub data: T,
    pub required_memory: usize,
}

impl<T: Clone> GPUBatchItem<T> {
    pub fn new(id: String, data: T, required_memory: usize) -> Self {
        Self {
            id,
            data,
            required_memory,
        }
    }
}

/// GPU batch request in the scheduler
#[derive(Clone, Debug)]
pub struct GPUBatchRequest<T: Clone> {
    pub items: Vec<GPUBatchItem<T>>,
    pub priority: u32,
    pub timestamp: std::time::Instant,
}

impl<T: Clone> GPUBatchRequest<T> {
    pub fn new(items: Vec<GPUBatchItem<T>>, priority: u32) -> Self {
        Self {
            items,
            priority,
            timestamp: std::time::Instant::now(),
        }
    }

    /// Calculate total memory required for this batch
    pub fn total_memory_required(&self) -> usize {
        self.items.iter().map(|item| item.required_memory).sum()
    }
}

/// GPU compute pipeline configuration
#[derive(Clone, Debug)]
pub struct GPUComputePipeline {
    pub shader_name: String,
    pub thread_group_size: (u32, u32, u32),
    pub features: Vec<String>,
}

impl GPUComputePipeline {
    pub fn new(shader_name: String, thread_group_size: (u32, u32, u32)) -> Self {
        Self {
            shader_name,
            thread_group_size,
            features: Vec::new(),
        }
    }

    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.features = features;
        self
    }

    /// Get recommended thread count for batch
    pub fn recommended_threads(&self, batch_size: usize) -> usize {
        let threads_per_group = self.thread_group_size.0 as usize
            * self.thread_group_size.1 as usize
            * self.thread_group_size.2 as usize;
        batch_size.div_ceil(threads_per_group) * threads_per_group
    }
}

/// GPU batch scheduler
pub struct GPUBatchScheduler {
    memory_pool: Arc<RwLock<GPUMemoryPool>>,
    pipeline: GPUComputePipeline,
    max_queue_size: usize,
    queue: Arc<RwLock<VecDeque<GPUBatchRequest<Vec<u8>>>>>,
}

impl GPUBatchScheduler {
    pub fn new(total_gpu_memory: usize, pipeline: GPUComputePipeline) -> Self {
        Self {
            memory_pool: Arc::new(RwLock::new(GPUMemoryPool::new(total_gpu_memory))),
            pipeline,
            max_queue_size: 100,
            queue: Arc::new(RwLock::new(VecDeque::new())),
        }
    }

    /// Schedule a batch request
    pub fn schedule_batch(&self, request: GPUBatchRequest<Vec<u8>>) -> Result<String, String> {
        let mut queue = self.queue.write();

        if queue.len() >= self.max_queue_size {
            return Err("Queue is full".to_string());
        }

        // Check if we have enough memory
        let required_memory = request.total_memory_required();
        let pool = self.memory_pool.read();
        let (used, total) = pool.memory_usage();

        if used + required_memory > total {
            return Err("Insufficient GPU memory".to_string());
        }

        drop(pool); // Release lock before inserting

        queue.push_back(request);
        Ok("Batch scheduled".to_string())
    }

    /// Get next batch from queue
    pub fn get_next_batch(&self) -> Option<GPUBatchRequest<Vec<u8>>> {
        let mut queue = self.queue.write();

        // Sort by priority (higher priority first), then by timestamp (FIFO for same priority)
        let mut items: Vec<_> = queue.drain(..).collect();
        items.sort_by(|a, b| match b.priority.cmp(&a.priority) {
            std::cmp::Ordering::Equal => a.timestamp.cmp(&b.timestamp),
            other => other,
        });

        if !items.is_empty() {
            let batch = items.remove(0); // Take the highest priority item
            for item in items {
                queue.push_back(item); // Put remaining items back
            }
            Some(batch)
        } else {
            None
        }
    }

    /// Get GPU memory usage
    pub fn get_memory_usage(&self) -> (usize, usize) {
        self.memory_pool.read().memory_usage()
    }

    /// Get GPU memory usage percentage
    pub fn get_memory_usage_percent(&self) -> f32 {
        self.memory_pool.read().memory_usage_percent()
    }

    /// Get queue size
    pub fn get_queue_size(&self) -> usize {
        self.queue.read().len()
    }

    /// Get compute pipeline
    pub fn get_pipeline(&self) -> &GPUComputePipeline {
        &self.pipeline
    }

    /// Allocate GPU memory
    pub fn allocate_memory(&self, size: usize) -> Option<GPUMemoryAllocation> {
        self.memory_pool.write().allocate(size)
    }

    /// Deallocate GPU memory
    pub fn deallocate_memory(&self, allocation: &GPUMemoryAllocation) {
        self.memory_pool.write().deallocate(allocation);
    }
}

/// GPU batch executor (mock for Phase 5)
pub struct GPUBatchExecutor {
    scheduler: Arc<GPUBatchScheduler>,
}

impl GPUBatchExecutor {
    pub fn new(scheduler: Arc<GPUBatchScheduler>) -> Self {
        Self { scheduler }
    }

    /// Execute a batch on GPU (mock implementation)
    pub fn execute_batch(&self, batch: GPUBatchRequest<Vec<u8>>) -> Result<Vec<u8>, String> {
        // Mock: simulate GPU processing time
        let processing_time = std::time::Duration::from_millis(
            (batch.items.len() as u64) * 5, // 5ms per item
        );
        std::thread::sleep(processing_time);

        // Mock: return concatenated output
        let output: Vec<u8> = batch
            .items
            .iter()
            .flat_map(|item| item.data.iter().cloned())
            .collect();

        Ok(output)
    }

    /// Get scheduler reference
    pub fn get_scheduler(&self) -> &GPUBatchScheduler {
        &self.scheduler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_memory_allocation() {
        let mut pool = GPUMemoryPool::new(1024);
        let alloc = pool.allocate(256);

        assert!(alloc.is_some());
        assert_eq!(pool.memory_usage().0, 256);
    }

    #[test]
    fn test_gpu_memory_insufficient() {
        let mut pool = GPUMemoryPool::new(256);
        let alloc = pool.allocate(512);

        assert!(alloc.is_none());
    }

    #[test]
    fn test_gpu_memory_deallocation() {
        let mut pool = GPUMemoryPool::new(1024);
        let alloc = pool.allocate(256).unwrap();
        pool.deallocate(&alloc);

        assert_eq!(pool.memory_usage().0, 0);
    }

    #[test]
    fn test_gpu_memory_usage_percent() {
        let mut pool = GPUMemoryPool::new(1000);
        pool.allocate(500).unwrap();

        let usage_percent = pool.memory_usage_percent();
        assert!(usage_percent > 49.0 && usage_percent < 51.0); // ~50%
    }

    #[test]
    fn test_gpu_batch_item_creation() {
        let item = GPUBatchItem::new("id1".to_string(), vec![1, 2, 3], 256);
        assert_eq!(item.id, "id1");
        assert_eq!(item.required_memory, 256);
    }

    #[test]
    fn test_gpu_batch_request_memory() {
        let items = vec![
            GPUBatchItem::new("id1".to_string(), vec![1, 2], 256),
            GPUBatchItem::new("id2".to_string(), vec![3, 4], 512),
        ];
        let request = GPUBatchRequest::new(items, 1);

        assert_eq!(request.total_memory_required(), 768);
    }

    #[test]
    fn test_gpu_scheduler_creation() {
        let pipeline = GPUComputePipeline::new("tokenizer".to_string(), (8, 8, 1));
        let scheduler = GPUBatchScheduler::new(8192, pipeline);

        assert_eq!(scheduler.get_memory_usage().1, 8192);
        assert_eq!(scheduler.get_queue_size(), 0);
    }

    #[test]
    fn test_gpu_scheduler_schedule_batch() {
        let pipeline = GPUComputePipeline::new("tokenizer".to_string(), (8, 8, 1));
        let scheduler = GPUBatchScheduler::new(8192, pipeline);

        let items = vec![GPUBatchItem::new("id1".to_string(), vec![1, 2], 256)];
        let request = GPUBatchRequest::new(items, 1);

        let result = scheduler.schedule_batch(request);
        assert!(result.is_ok());
        assert_eq!(scheduler.get_queue_size(), 1);
    }

    #[test]
    fn test_gpu_scheduler_insufficient_memory() {
        let pipeline = GPUComputePipeline::new("tokenizer".to_string(), (8, 8, 1));
        let scheduler = GPUBatchScheduler::new(256, pipeline); // Small memory

        let items = vec![GPUBatchItem::new("id1".to_string(), vec![0; 512], 512)];
        let request = GPUBatchRequest::new(items, 1);

        let result = scheduler.schedule_batch(request);
        assert!(result.is_err());
    }

    #[test]
    fn test_gpu_batch_executor_execution() {
        let pipeline = GPUComputePipeline::new("tokenizer".to_string(), (8, 8, 1));
        let scheduler = Arc::new(GPUBatchScheduler::new(8192, pipeline));
        let executor = GPUBatchExecutor::new(scheduler);

        let items = vec![GPUBatchItem::new("id1".to_string(), vec![1, 2, 3], 256)];
        let request = GPUBatchRequest::new(items, 1);

        let result = executor.execute_batch(request);
        assert!(result.is_ok());
    }

    #[test]
    fn test_gpu_compute_pipeline() {
        let pipeline = GPUComputePipeline::new("tokenizer".to_string(), (256, 1, 1));
        let recommended_threads = pipeline.recommended_threads(512);

        assert!(recommended_threads >= 512);
    }

    #[test]
    fn test_gpu_scheduler_priority_queue() {
        let pipeline = GPUComputePipeline::new("tokenizer".to_string(), (8, 8, 1));
        let scheduler = GPUBatchScheduler::new(8192, pipeline);

        let items1 = vec![GPUBatchItem::new("id1".to_string(), vec![1], 100)];
        let request1 = GPUBatchRequest::new(items1, 1); // Low priority

        let items2 = vec![GPUBatchItem::new("id2".to_string(), vec![2], 100)];
        let request2 = GPUBatchRequest::new(items2, 10); // High priority

        let _ = scheduler.schedule_batch(request1);
        let _ = scheduler.schedule_batch(request2);

        // Next batch should be the high-priority one
        if let Some(batch) = scheduler.get_next_batch() {
            assert_eq!(batch.items[0].id, "id2");
        }
    }
}
