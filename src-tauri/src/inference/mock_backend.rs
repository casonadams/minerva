/// Mock Backend for Testing
///
/// Provides a mock implementation of the InferenceBackend trait
/// for testing and development without requiring actual model files.
/// Generates intelligent mock responses based on prompt content.
use crate::error::MinervaResult;
use crate::inference::inference_backend_trait::{GenerationParams, InferenceBackend};
use std::path::Path;

/// Mock backend for testing and development
#[derive(Debug)]
#[allow(dead_code)]
pub struct MockBackend {
    loaded: bool,
    n_ctx: usize,
    n_threads: usize,
}

impl MockBackend {
    /// Create new mock backend
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            loaded: false,
            n_ctx: 0,
            n_threads: num_cpus::get(),
        }
    }

    /// Generate intelligent mock response based on prompt
    ///
    /// Simulates different response patterns for different question types:
    /// - Greeting questions return friendly responses
    /// - What/How questions return multi-sentence explanations
    /// - Why questions return reasoned explanations
    /// - Explain/Describe requests return detailed descriptions
    /// - Other prompts return thoughtful generic responses
    fn generate_intelligent_response(&self, prompt: &str, max_tokens: usize) -> String {
        let prompt_lower = prompt.to_lowercase();

        let base = if prompt_lower.contains("hello") || prompt_lower.contains("hi") {
            "Hello! I'm an AI assistant. How can I help you today?"
        } else if prompt_lower.contains("what") || prompt_lower.contains("how") {
            "That's an interesting question. Let me provide a thoughtful response. \
             The answer involves multiple interconnected factors. \
             First, consider the foundational principles. \
             Then, examine the practical implications. \
             This comprehensive approach provides better understanding."
        } else if prompt_lower.contains("why") {
            "There are several compelling reasons for this. \
             The primary reason relates to natural efficiency patterns. \
             Historical precedent supports this approach. \
             Contemporary research confirms these findings."
        } else if prompt_lower.contains("explain") || prompt_lower.contains("describe") {
            "Let me provide a detailed explanation. \
             This topic encompasses several key components. \
             Understanding requires examining foundational concepts. \
             Advanced aspects build upon this foundation. \
             This systematic approach ensures comprehensive understanding."
        } else {
            "That's an interesting question. \
             Let me provide a thoughtful analysis. \
             This involves examining multiple perspectives. \
             Different viewpoints offer valuable insights. \
             We should consider both theory and practice."
        };

        // Truncate to max_tokens (approximate as words)
        let words: Vec<&str> = base.split_whitespace().collect();
        if words.len() > max_tokens {
            words[..max_tokens].join(" ")
        } else {
            base.to_string()
        }
    }
}

impl Default for MockBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for MockBackend {
    fn load_model(&mut self, path: &Path, n_ctx: usize) -> MinervaResult<()> {
        if !path.exists() {
            return Err(crate::error::MinervaError::ModelNotFound(format!(
                "Model not found: {}",
                path.display()
            )));
        }
        self.loaded = true;
        self.n_ctx = n_ctx;
        Ok(())
    }

    fn unload_model(&mut self) {
        self.loaded = false;
    }

    fn generate(&self, prompt: &str, params: GenerationParams) -> MinervaResult<String> {
        if !self.loaded {
            return Err(crate::error::MinervaError::InferenceError(
                "Model not loaded".to_string(),
            ));
        }

        // Simulate real inference
        std::thread::sleep(std::time::Duration::from_millis(50));

        let response = self.generate_intelligent_response(prompt, params.max_tokens);
        Ok(response)
    }

    fn tokenize(&self, text: &str) -> MinervaResult<Vec<i32>> {
        // Simple word-based mock tokenization
        Ok(text
            .split_whitespace()
            .enumerate()
            .map(|(i, _)| i as i32)
            .collect())
    }

    fn detokenize(&self, tokens: &[i32]) -> MinervaResult<String> {
        // Mock detokenization
        Ok(format!("[{} tokens]", tokens.len()))
    }

    fn is_loaded(&self) -> bool {
        self.loaded
    }

    fn context_size(&self) -> usize {
        self.n_ctx
    }

    fn thread_count(&self) -> usize {
        self.n_threads
    }
}
