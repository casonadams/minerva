use crate::error::{MinervaError, MinervaResult};
/// Helper functions for GGUF tensor extraction
use ndarray::{Array1, Array2};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;

/// Load all SafeTensors files from a directory
pub fn load_safetensors_files(path: &Path) -> MinervaResult<HashMap<String, Vec<u8>>> {
    // First check if this is a single safetensors file
    if path.extension().map_or(false, |ext| ext == "safetensors") {
        return load_single_safetensors(path);
    }

    // Otherwise, load from directory
    let index_path = path.join("model.safetensors.index.json");

    if !index_path.exists() {
        // Try single model.safetensors
        let single_path = path.join("model.safetensors");
        if single_path.exists() {
            return load_single_safetensors(&single_path);
        }

        return Err(MinervaError::ModelLoadingError(format!(
            "No model.safetensors.index.json or model.safetensors found at {:?}",
            path
        )));
    }

    let index_content = std::fs::read_to_string(&index_path)
        .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to read index: {}", e)))?;

    let index: serde_json::Value = serde_json::from_str(&index_content)?;

    let weights_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| MinervaError::ModelLoadingError("Invalid index.json format".into()))?;

    // Find unique shard files
    let mut shard_files = std::collections::HashSet::new();
    for shard in weights_map.values() {
        if let Some(s) = shard.as_str() {
            shard_files.insert(s.to_string());
        }
    }

    // Load all shards
    let mut all_tensors: HashMap<String, Vec<u8>> = HashMap::new();

    for shard_file in shard_files {
        let shard_path = path.join(&shard_file);
        let shard_tensors = load_single_safetensors(&shard_path)?;
        all_tensors.extend(shard_tensors);
    }

    Ok(all_tensors)
}

/// Load a single SafeTensors file
pub fn load_single_safetensors(path: &Path) -> MinervaResult<HashMap<String, Vec<u8>>> {
    let bytes = std::fs::read(path).map_err(|e| {
        MinervaError::ModelLoadingError(format!("Failed to read {}: {}", path.display(), e))
    })?;

    let st = SafeTensors::deserialize(&bytes)
        .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to deserialize: {}", e)))?;

    let mut result = HashMap::new();
    for tensor_name in st.names() {
        match st.tensor(tensor_name) {
            Ok(tensor) => {
                result.insert(tensor_name.to_string(), tensor.data().to_vec());
            }
            Err(e) => {
                eprintln!("Warning: Failed to load tensor {}: {}", tensor_name, e);
            }
        }
    }

    Ok(result)
}

/// Extract a 2D tensor from the tensor map
pub fn extract_tensor_2d(
    tensors: &HashMap<String, Vec<u8>>,
    name: &str,
) -> MinervaResult<Array2<f32>> {
    let data = tensors
        .get(name)
        .ok_or_else(|| MinervaError::ModelLoadingError(format!("Missing tensor: {}", name)))?;

    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    let size = floats.len();

    // Common shapes for GPT-OSS 20B
    let shape = if size == 201088 * 2880 {
        Some((201088, 2880))
    } else if size == 2880 * 2880 {
        Some((2880, 2880))
    } else if size == 2880 * 7168 {
        Some((2880, 7168))
    } else if size == 7168 * 2880 {
        Some((7168, 2880))
    } else if size == 2880 * 768 {
        Some((2880, 768))
    } else {
        let sqrt = (size as f64).sqrt() as usize;
        if sqrt * sqrt == size {
            Some((sqrt, sqrt))
        } else {
            for i in (1..=1000).rev() {
                if size % i == 0 {
                    return Array2::from_shape_vec((size / i, i), floats).map_err(|e| {
                        MinervaError::ModelLoadingError(format!("Failed to create array: {}", e))
                    });
                }
            }
            None
        }
    };

    let final_shape = shape.ok_or_else(|| {
        MinervaError::ModelLoadingError("Failed to determine tensor shape".into())
    })?;

    Array2::from_shape_vec(final_shape, floats)
        .map_err(|e| MinervaError::ModelLoadingError(format!("Failed to create array: {}", e)))
}

/// Extract a 1D tensor from the tensor map
pub fn extract_tensor_1d(
    tensors: &HashMap<String, Vec<u8>>,
    name: &str,
) -> MinervaResult<Array1<f32>> {
    let data = tensors
        .get(name)
        .ok_or_else(|| MinervaError::ModelLoadingError(format!("Missing tensor: {}", name)))?;

    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(Array1::from_vec(floats))
}
