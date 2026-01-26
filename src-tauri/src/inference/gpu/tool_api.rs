/// Tool-Friendly API for AI Coding Agents
///
/// Minimal, efficient API designed for OpenCode.ai and similar tools
/// - Single function calls for common operations
/// - Compact JSON output for context efficiency
/// - Early return with partial data when possible
/// - No unnecessary allocations
use super::tool_optimized_loader::{QuickConfig, ToolOptimizedLoader};
use crate::error::MinervaResult;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Ultra-compact model info (fits in one line)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Format: "MODEL|QUANT|SIZE_MB|TENSORS|LAYERS|VOCAB"
    pub id: String,
    /// Time to load metadata (ms)
    pub load_ms: u32,
    /// Is loadable on current system
    pub loadable: bool,
}

impl ModelInfo {
    /// Parse compact ID back to components
    pub fn parse(&self) -> ParsedModelInfo {
        let parts: Vec<&str> = self.id.split('|').collect();
        if parts.len() == 6 {
            ParsedModelInfo {
                name: parts[0].to_string(),
                quant: parts[1].to_string(),
                size_mb: parts[2].parse().unwrap_or(0),
                tensors: parts[3].parse().unwrap_or(0),
                layers: parts[4].parse().unwrap_or(0),
                vocab: parts[5].parse().unwrap_or(0),
            }
        } else {
            ParsedModelInfo::default()
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ParsedModelInfo {
    pub name: String,
    pub quant: String,
    pub size_mb: u32,
    pub tensors: usize,
    pub layers: usize,
    pub vocab: usize,
}

/// Tool-optimized model loader API
pub struct ToolAPI;

impl ToolAPI {
    /// Get model info in <100ms
    /// Returns: {"id": "MODEL|QUANT|SIZE|TENSORS|LAYERS|VOCAB", "loadable": true, "load_ms": 45}
    pub fn get_model_info(path: &Path) -> MinervaResult<ModelInfo> {
        let start = std::time::Instant::now();

        let loader = ToolOptimizedLoader::quick_load(path)?;
        let load_ms = start.elapsed().as_millis() as u32;

        Ok(ModelInfo {
            id: loader.summary(),
            load_ms,
            loadable: loader.is_loadable(),
        })
    }

    /// Batch check multiple models - returns compact info for each
    pub fn check_models(paths: &[&Path]) -> Vec<(String, Result<ModelInfo, String>)> {
        paths
            .iter()
            .map(|path| {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("unknown")
                    .to_string();
                let result = Self::get_model_info(path).map_err(|e| format!("{}", e));
                (name, result)
            })
            .collect()
    }

    /// Minimal config needed for inference planning
    pub fn get_inference_config(path: &Path) -> MinervaResult<QuickConfig> {
        let loader = ToolOptimizedLoader::quick_load(path)?;
        Ok(loader.config)
    }

    /// Estimate throughput based on model size
    pub fn estimate_throughput(config: &QuickConfig) -> EstimatedThroughput {
        let total_params = config.hidden_size * config.num_layers * 2;
        let estimated_flops = total_params as f64 * 2.0; // Rough estimate

        EstimatedThroughput {
            tokens_per_second_single: (10.0 / (config.hidden_size as f64 / 2880.0)) as u32,
            tokens_per_second_batch_20: (300.0 / (config.hidden_size as f64 / 2880.0)) as u32,
            estimated_flops_per_token: estimated_flops as u64,
            notes: "Estimates for MXFP4 quantized model".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatedThroughput {
    pub tokens_per_second_single: u32,
    pub tokens_per_second_batch_20: u32,
    pub estimated_flops_per_token: u64,
    pub notes: String,
}

/// Compact format for tool communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactModelSpec {
    /// "gpt-oss-20b|gguf|12110|459|24|201088"
    pub spec: String,
    /// true if loadable
    pub ok: bool,
}

impl CompactModelSpec {
    pub fn new(spec: String, ok: bool) -> Self {
        Self { spec, ok }
    }

    pub fn from_info(info: &ModelInfo) -> Self {
        Self {
            spec: info.id.clone(),
            ok: info.loadable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_parse() {
        let info = ModelInfo {
            id: "GPT-OSS-20B|MXFP4|12110|459|24|201088".to_string(),
            load_ms: 45,
            loadable: true,
        };

        let parsed = info.parse();
        assert_eq!(parsed.name, "GPT-OSS-20B");
        assert_eq!(parsed.quant, "MXFP4");
        assert_eq!(parsed.size_mb, 12110);
        assert_eq!(parsed.tensors, 459);
        assert_eq!(parsed.layers, 24);
        assert_eq!(parsed.vocab, 201088);
    }

    #[test]
    fn test_estimate_throughput() {
        let config = QuickConfig {
            hidden_size: 2880,
            num_layers: 24,
            vocab_size: 201088,
            max_seq_len: 4096,
        };

        let throughput = ToolAPI::estimate_throughput(&config);
        assert!(throughput.tokens_per_second_single > 0);
        assert!(throughput.tokens_per_second_batch_20 > 0);
    }

    #[test]
    fn test_compact_spec() {
        let spec = CompactModelSpec::new("GPT-OSS-20B|MXFP4|12110|459|24|201088".to_string(), true);
        assert_eq!(spec.spec.split('|').count(), 6);
    }
}
