/// Token Decoding and Generation
///
/// Provides token sampling and sequence generation using various sampling strategies:
/// - Greedy: Always select highest probability token
/// - Top-K: Sample from k most likely tokens
/// - Top-P: Sample from tokens with cumulative probability p
use crate::error::{MinervaError, MinervaResult};

/// Token sampling strategy
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Greedy decoding - always pick highest probability token
    Greedy,
    /// Top-k sampling - sample from k most likely tokens
    TopK(usize),
    /// Top-p (nucleus) sampling - sample from tokens with cumulative probability p
    TopP(f32),
}

/// Parameters for token sampling
pub struct SamplingParams {
    /// Temperature for controlling randomness
    pub temperature: f32,
    /// Sampling strategy
    pub strategy: SamplingStrategy,
}

impl SamplingParams {
    /// Create new sampling params with greedy strategy
    pub fn greedy(temperature: f32) -> Self {
        Self {
            temperature,
            strategy: SamplingStrategy::Greedy,
        }
    }
}

/// Parameters for TokenGenerator::generate
pub struct GenerationParams<'a> {
    /// Initial tokens to start generation
    pub initial_tokens: &'a [usize],
    /// Number of tokens to generate
    pub num_tokens: usize,
    /// Sampling parameters
    pub sampling: SamplingParams,
}

/// Decoder for token generation
pub struct Decoder {
    vocab_size: usize,
    max_seq_len: usize,
}

impl Decoder {
    /// Create new decoder
    pub fn new(vocab_size: usize, max_seq_len: usize) -> Self {
        Self {
            vocab_size,
            max_seq_len,
        }
    }

    /// Sample next token from logits
    pub fn sample_token(&self, logits: &[f32], params: SamplingParams) -> MinervaResult<usize> {
        if logits.len() != self.vocab_size {
            return Err(MinervaError::InferenceError(format!(
                "Logits size {} != vocab size {}",
                logits.len(),
                self.vocab_size
            )));
        }

        if params.temperature <= 0.0 {
            return Err(MinervaError::InferenceError(
                "Temperature must be positive".to_string(),
            ));
        }

        // Apply temperature scaling
        let probs = logits
            .iter()
            .map(|l| l / params.temperature)
            .collect::<Vec<_>>();

        // Apply softmax
        let max = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = probs.iter().map(|p| (p - max).exp()).collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Apply sampling strategy
        let token = match params.strategy {
            SamplingStrategy::Greedy => probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| MinervaError::InferenceError("No valid token found".to_string()))?,

            SamplingStrategy::TopK(k) => {
                if k == 0 {
                    return Err(MinervaError::InferenceError("k must be > 0".to_string()));
                }
                let k = k.min(self.vocab_size);
                let mut indices: Vec<_> = (0..self.vocab_size).collect();
                indices.sort_by(|a, b| {
                    probs[*b]
                        .partial_cmp(&probs[*a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                // Zero out probabilities outside top-k
                for i in k..self.vocab_size {
                    probs[indices[i]] = 0.0;
                }

                // Renormalize
                let sum: f32 = probs.iter().sum();
                if sum > 0.0 {
                    for p in &mut probs {
                        *p /= sum;
                    }
                }

                // Sample from top-k
                self.sample_categorical(&probs)?
            }

            SamplingStrategy::TopP(p) => {
                if p <= 0.0 || p > 1.0 {
                    return Err(MinervaError::InferenceError(
                        "p must be in (0, 1]".to_string(),
                    ));
                }

                let mut indices: Vec<_> = (0..self.vocab_size).collect();
                indices.sort_by(|a, b| {
                    probs[*b]
                        .partial_cmp(&probs[*a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let mut cumsum = 0.0;
                for i in indices.iter().cloned() {
                    cumsum += probs[i];
                    if cumsum < p {
                        // Keep this token
                    } else {
                        probs[i] = 0.0;
                    }
                }

                // Renormalize
                let sum: f32 = probs.iter().sum();
                if sum > 0.0 {
                    for p in &mut probs {
                        *p /= sum;
                    }
                }

                self.sample_categorical(&probs)?
            }
        };

        Ok(token)
    }

    /// Sample from categorical distribution
    fn sample_categorical(&self, probs: &[f32]) -> MinervaResult<usize> {
        let mut cumsum = 0.0;
        let rand = (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as f32)
            / 1e9;
        let rand = rand.fract();

        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if rand < cumsum {
                return Ok(i);
            }
        }

        // Return last token if rounding errors
        Ok(probs.len().saturating_sub(1))
    }

    /// Generate tokens
    pub fn generate(
        &self,
        params: GenerationParams,
        mut forward: impl FnMut(&[usize]) -> MinervaResult<Vec<f32>>,
    ) -> MinervaResult<Vec<usize>> {
        if params.initial_tokens.is_empty() {
            return Err(MinervaError::InferenceError(
                "Initial tokens cannot be empty".to_string(),
            ));
        }

        if params.initial_tokens.len() + params.num_tokens > self.max_seq_len {
            return Err(MinervaError::InferenceError(
                "Sequence too long for max_seq_len".to_string(),
            ));
        }

        let mut tokens = params.initial_tokens.to_vec();
        let mut sequence = params.initial_tokens.to_vec();

        for _ in 0..params.num_tokens {
            let logits = forward(&tokens)?;
            let sampling = SamplingParams {
                temperature: params.sampling.temperature,
                strategy: params.sampling.strategy,
            };
            let next_token = self.sample_token(&logits, sampling)?;
            tokens.push(next_token);
            sequence.push(next_token);
        }

        Ok(sequence)
    }
}
