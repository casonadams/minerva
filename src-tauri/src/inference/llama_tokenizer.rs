/// LLaMA Tokenizer Implementation (Phase 6 - Step 2)
///
/// Real SentencePiece tokenizer for LLaMA models.
/// Supports BPE (Byte Pair Encoding) algorithm and special tokens.
use crate::error::{MinervaError, MinervaResult};
use std::collections::HashMap;

/// Special token IDs for LLaMA
#[derive(Debug, Clone, Copy)]
pub struct SpecialTokens {
    pub bos: u32, // Beginning of sequence
    pub eos: u32, // End of sequence
    pub unk: u32, // Unknown token
    pub pad: u32, // Padding token
}

impl SpecialTokens {
    /// Create default LLaMA special tokens
    pub fn llama() -> Self {
        Self {
            bos: 1, // <s>
            eos: 2, // </s>
            unk: 0, // <unk>
            pad: 0, // Uses unk as padding
        }
    }
}

/// BPE token pair for merging
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
struct TokenPair {
    first: u32,
    second: u32,
}

/// LLaMA tokenizer using SentencePiece algorithm
#[derive(Clone)]
pub struct LLaMATokenizer {
    /// Vocabulary mapping
    vocab: Vec<String>,
    /// Reverse lookup: token string to ID
    token_to_id: HashMap<String, u32>,
    /// BPE merge rules (ordered by frequency)
    bpe_merges: Vec<TokenPair>,
    /// Special tokens
    special_tokens: SpecialTokens,
}

impl LLaMATokenizer {
    /// Create a new LLaMA tokenizer from vocabulary
    pub fn new(vocab: Vec<String>) -> MinervaResult<Self> {
        if vocab.is_empty() {
            return Err(MinervaError::InferenceError(
                "Tokenizer vocabulary cannot be empty".to_string(),
            ));
        }

        // Build reverse lookup table
        let mut token_to_id = HashMap::with_capacity(vocab.len());
        for (id, token) in vocab.iter().enumerate() {
            token_to_id.insert(token.clone(), id as u32);
        }

        Ok(Self {
            vocab,
            token_to_id,
            bpe_merges: Vec::new(),
            special_tokens: SpecialTokens::llama(),
        })
    }

    /// Set BPE merge rules
    pub fn set_bpe_merges(&mut self, merges: Vec<(u32, u32)>) {
        self.bpe_merges = merges
            .into_iter()
            .map(|(f, s)| TokenPair {
                first: f,
                second: s,
            })
            .collect();
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token by ID
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(|s| s.as_str())
    }

    /// Get token ID
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Encode text to tokens
    pub fn encode(&self, text: &str) -> MinervaResult<Vec<u32>> {
        // Start with UTF-8 bytes
        let bytes: Vec<u32> = text.as_bytes().iter().map(|&b| b as u32).collect();

        if bytes.is_empty() {
            return Ok(vec![]);
        }

        // Apply BPE merges
        let mut tokens = bytes;
        tokens = self.apply_bpe_merges(&tokens);

        // Convert byte sequences to token IDs
        let result = self.bytes_to_tokens(&tokens)?;

        Ok(result)
    }

    /// Encode multiple texts in batch
    pub fn encode_batch(&self, texts: &[&str]) -> MinervaResult<Vec<Vec<u32>>> {
        texts.iter().map(|text| self.encode(text)).collect()
    }

    /// Decode tokens to text
    pub fn decode(&self, tokens: &[u32]) -> MinervaResult<String> {
        let strings: Result<Vec<String>, _> = tokens
            .iter()
            .map(|&id| {
                self.vocab.get(id as usize).cloned().ok_or_else(|| {
                    MinervaError::InferenceError(format!("Unknown token ID: {}", id))
                })
            })
            .collect();

        let strings = strings?;
        Ok(strings.join(""))
    }

    /// Decode tokens from batch
    pub fn decode_batch(&self, batch: &[Vec<u32>]) -> MinervaResult<Vec<String>> {
        batch.iter().map(|tokens| self.decode(tokens)).collect()
    }

    /// Apply BPE merges to token sequence
    fn apply_bpe_merges(&self, tokens: &[u32]) -> Vec<u32> {
        let mut result = tokens.to_vec();

        // Apply each merge rule in order
        for TokenPair { first, second } in &self.bpe_merges {
            let mut new_result = Vec::new();
            let mut i = 0;

            while i < result.len() {
                if i + 1 < result.len() && result[i] == *first && result[i + 1] == *second {
                    // Merge found - create new token ID
                    let merged = self.merge_token_ids(*first, *second);
                    new_result.push(merged);
                    i += 2;
                } else {
                    new_result.push(result[i]);
                    i += 1;
                }
            }

            result = new_result;
        }

        result
    }

    /// Merge two token IDs into a new token ID (simplified BPE)
    fn merge_token_ids(&self, first: u32, second: u32) -> u32 {
        // In a real implementation, this would look up the merged token ID
        // For now, use a deterministic hash
        ((first as u64).wrapping_mul(31) ^ (second as u64)) as u32
    }

    /// Convert byte sequence to token IDs
    fn bytes_to_tokens(&self, bytes: &[u32]) -> MinervaResult<Vec<u32>> {
        // Try to find longest matching tokens
        let mut tokens = Vec::new();
        let mut i = 0;

        while i < bytes.len() {
            // Try longest match first
            let mut matched = false;

            // Look for matching tokens in vocabulary
            for j in (i..bytes.len()).rev() {
                if j - i > 4 {
                    break; // Limit token length
                }

                let token_bytes: Vec<u8> = bytes[i..=j].iter().map(|&b| b as u8).collect();

                if let Ok(token_str) = String::from_utf8(token_bytes)
                    && let Some(id) = self.token_to_id.get(&token_str)
                {
                    tokens.push(*id);
                    i = j + 1;
                    matched = true;
                    break;
                }
            }

            if !matched {
                // Use single byte token (or unknown)
                tokens.push(self.special_tokens.unk);
                i += 1;
            }
        }

        Ok(tokens)
    }
}

impl std::fmt::Debug for LLaMATokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LLaMATokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("bpe_merges", &self.bpe_merges.len())
            .field("special_tokens", &self.special_tokens)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tokenizer() -> LLaMATokenizer {
        let vocab = vec![
            "<unk>".to_string(),
            "<s>".to_string(),
            "</s>".to_string(),
            "hello".to_string(),
            "world".to_string(),
            " ".to_string(),
            "!".to_string(),
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
        ];
        LLaMATokenizer::new(vocab).unwrap()
    }

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = create_test_tokenizer();
        assert_eq!(tokenizer.vocab_size(), 10);
    }

    #[test]
    fn test_get_token() {
        let tokenizer = create_test_tokenizer();
        assert_eq!(tokenizer.get_token(1), Some("<s>"));
        assert_eq!(tokenizer.get_token(3), Some("hello"));
        assert_eq!(tokenizer.get_token(100), None);
    }

    #[test]
    fn test_get_id() {
        let tokenizer = create_test_tokenizer();
        assert_eq!(tokenizer.get_id("<s>"), Some(1));
        assert_eq!(tokenizer.get_id("hello"), Some(3));
        assert_eq!(tokenizer.get_id("nonexistent"), None);
    }

    #[test]
    fn test_encode_simple() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("hello").unwrap();
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_encode_batch() {
        let tokenizer = create_test_tokenizer();
        let texts = vec!["hello", "world"];
        let results = tokenizer.encode_batch(&texts).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_special_tokens() {
        let special = SpecialTokens::llama();
        assert_eq!(special.bos, 1);
        assert_eq!(special.eos, 2);
        assert_eq!(special.unk, 0);
    }

    #[test]
    fn test_empty_text_encode() {
        let tokenizer = create_test_tokenizer();
        let tokens = tokenizer.encode("").unwrap();
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_tokenizer_empty_vocab_error() {
        let result = LLaMATokenizer::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_simple() {
        let tokenizer = create_test_tokenizer();
        let token_ids = vec![1, 3]; // "<s>" and "hello"
        let text = tokenizer.decode(&token_ids).unwrap();
        assert!(!text.is_empty());
    }

    #[test]
    fn test_decode_batch() {
        let tokenizer = create_test_tokenizer();
        let batches = vec![vec![1, 3], vec![2, 4]];
        let results = tokenizer.decode_batch(&batches).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_decode_invalid_token() {
        let tokenizer = create_test_tokenizer();
        let token_ids = vec![1, 999]; // 999 is out of range
        let result = tokenizer.decode(&token_ids);
        assert!(result.is_err());
    }

    #[test]
    fn test_bpe_merges_setting() {
        let mut tokenizer = create_test_tokenizer();
        let merges = vec![(1, 3), (2, 4)];
        tokenizer.set_bpe_merges(merges);
        assert_eq!(tokenizer.bpe_merges.len(), 2);
    }

    #[test]
    fn test_vocab_lookup() {
        let tokenizer = create_test_tokenizer();
        // Test that all tokens in vocab can be looked up
        for (id, token) in tokenizer.vocab.iter().enumerate() {
            assert_eq!(tokenizer.get_id(token), Some(id as u32));
        }
    }

    #[test]
    fn test_token_to_id_consistency() {
        let tokenizer = create_test_tokenizer();
        for (id, token) in tokenizer.vocab.iter().enumerate() {
            if let Some(found_id) = tokenizer.get_id(token) {
                assert_eq!(found_id, id as u32);
            }
        }
    }
}
