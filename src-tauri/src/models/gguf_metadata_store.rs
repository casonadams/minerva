use super::gguf_loader::GGUFModelMetadata;

/// Stores metadata values from GGUF key-value pairs
pub struct GGUFMetadataStore;

impl GGUFMetadataStore {
    /// Store u32 metadata value
    pub fn store_u32(key: &str, value: u32, metadata: &mut GGUFModelMetadata) {
        match key {
            "llama.context_length" => metadata.context_window = Some(value as usize),
            "llama.embedding_length" => metadata.embedding_length = Some(value as usize),
            "llama.feed_forward_length" => metadata.feed_forward_length = Some(value as usize),
            "llama.attention.head_count" => metadata.attention_head_count = Some(value as usize),
            "llama.attention.head_count_kv" => {
                metadata.attention_head_count_kv = Some(value as usize);
            }
            "llama.block_count" => metadata.layer_count = Some(value as usize),
            "quantization_version" => metadata.quantization_version = Some(value as usize),
            _ => {}
        }
    }

    /// Store i32 metadata value
    pub fn store_i32(key: &str, value: i32, metadata: &mut GGUFModelMetadata) {
        match key {
            "llama.context_length" => metadata.context_window = Some(value as usize),
            "llama.embedding_length" => metadata.embedding_length = Some(value as usize),
            "llama.feed_forward_length" => metadata.feed_forward_length = Some(value as usize),
            "llama.attention.head_count" => metadata.attention_head_count = Some(value as usize),
            "llama.attention.head_count_kv" => {
                metadata.attention_head_count_kv = Some(value as usize);
            }
            "llama.block_count" => metadata.layer_count = Some(value as usize),
            _ => {}
        }
    }

    /// Store u64 metadata value
    pub fn store_u64(key: &str, value: u64, metadata: &mut GGUFModelMetadata) {
        match key {
            "llama.context_length" => metadata.context_window = Some(value as usize),
            "llama.embedding_length" => metadata.embedding_length = Some(value as usize),
            "llama.feed_forward_length" => metadata.feed_forward_length = Some(value as usize),
            "llama.attention.head_count" => metadata.attention_head_count = Some(value as usize),
            "llama.attention.head_count_kv" => {
                metadata.attention_head_count_kv = Some(value as usize);
            }
            "llama.block_count" => metadata.layer_count = Some(value as usize),
            _ => {}
        }
    }

    /// Store string metadata value
    pub fn store_string(key: &str, value: &str, metadata: &mut GGUFModelMetadata) {
        match key {
            "general.name" => metadata.name = Some(value.to_string()),
            "llama.model_name" => metadata.name = Some(value.to_string()),
            "llama.architecture" => metadata.architecture = Some(value.to_string()),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_u32_context_length() {
        let mut metadata = GGUFModelMetadata {
            name: None,
            architecture: None,
            context_window: None,
            embedding_length: None,
            feed_forward_length: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            layer_count: None,
            quantization_version: None,
        };
        GGUFMetadataStore::store_u32("llama.context_length", 2048, &mut metadata);
        assert_eq!(metadata.context_window, Some(2048));
    }

    #[test]
    fn test_store_string_name() {
        let mut metadata = GGUFModelMetadata {
            name: None,
            architecture: None,
            context_window: None,
            embedding_length: None,
            feed_forward_length: None,
            attention_head_count: None,
            attention_head_count_kv: None,
            layer_count: None,
            quantization_version: None,
        };
        GGUFMetadataStore::store_string("general.name", "Mistral", &mut metadata);
        assert_eq!(metadata.name, Some("Mistral".to_string()));
    }
}
