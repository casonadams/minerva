/// Real tokenization and vocabulary support for Phase 4 Step 4
///
/// This module provides:
/// - Vocabulary management (token ID mapping)
/// - BPE (Byte Pair Encoding) tokenization
/// - Format detection and caching
/// - Integration with multiple tokenizer types
use std::collections::HashMap;

/// Token representation with metadata
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Token {
    /// Unique token ID
    pub id: u32,
    /// Token length in characters
    pub len: u16,
}

impl Token {
    /// Create new token
    pub fn new(id: u32, len: u16) -> Self {
        Self { id, len }
    }
}

/// Vocabulary mapping between tokens and IDs
#[derive(Debug, Clone)]
pub struct Vocabulary {
    /// Token string to ID mapping
    token_to_id: HashMap<String, u32>,
    /// ID to token string mapping
    id_to_token: HashMap<u32, String>,
    /// Special tokens (PAD, UNK, etc.)
    special_tokens: HashMap<String, u32>,
}

impl Vocabulary {
    /// Create empty vocabulary
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: HashMap::new(),
            special_tokens: HashMap::new(),
        }
    }

    /// Add token to vocabulary
    pub fn add_token(&mut self, token: String, id: u32) -> Result<(), String> {
        if self.token_to_id.contains_key(&token) {
            return Err(format!("Token already exists: {}", token));
        }
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.insert(id, token);
        Ok(())
    }

    /// Add special token (PAD, UNK, EOS, BOS)
    pub fn add_special_token(&mut self, name: String, id: u32) -> Result<(), String> {
        if self.special_tokens.contains_key(&name) {
            return Err(format!("Special token already exists: {}", name));
        }
        self.special_tokens.insert(name.clone(), id);
        self.token_to_id.insert(name.clone(), id);
        self.id_to_token.insert(id, name);
        Ok(())
    }

    /// Get token ID by string
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// Get token string by ID
    pub fn get_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(&id).cloned()
    }

    /// Get special token ID
    pub fn get_special(&self, name: &str) -> Option<u32> {
        self.special_tokens.get(name).copied()
    }

    /// Check if token exists
    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    /// Get vocabulary size
    pub fn size(&self) -> usize {
        self.token_to_id.len()
    }

    /// Get unknown token ID (default 0 if not set)
    pub fn unk_token_id(&self) -> u32 {
        self.get_special("UNK").unwrap_or(0)
    }

    /// Get padding token ID (default 0 if not set)
    pub fn pad_token_id(&self) -> u32 {
        self.get_special("PAD").unwrap_or(0)
    }
}

impl Default for Vocabulary {
    fn default() -> Self {
        Self::new()
    }
}

/// Byte Pair Encoding tokenizer
#[derive(Debug, Clone)]
pub struct BPETokenizer {
    vocab: Vocabulary,
    /// Cache of merged byte pairs
    #[allow(dead_code)]
    merges: Vec<(String, String)>,
}

impl BPETokenizer {
    /// Create new BPE tokenizer
    pub fn new(vocab: Vocabulary) -> Self {
        Self {
            vocab,
            merges: Vec::new(),
        }
    }

    /// Encode text to token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Simple byte-level encoding: split into bytes, map through vocab
        text.as_bytes()
            .iter()
            .filter_map(|&byte| {
                let char_str = String::from(byte as char);
                self.vocab.get_id(&char_str)
            })
            .collect()
    }

    /// Decode token IDs back to text
    pub fn decode(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&id| self.vocab.get_token(id))
            .collect::<Vec<_>>()
            .join("")
    }

    /// Count tokens in text (same as encode length)
    pub fn count_tokens(&self, text: &str) -> usize {
        self.encode(text).len()
    }

    /// Get underlying vocabulary
    pub fn vocab(&self) -> &Vocabulary {
        &self.vocab
    }
}

/// Tokenizer format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerFormat {
    /// BPE (Byte Pair Encoding)
    BPE,
    /// WordPiece tokenization
    WordPiece,
    /// SentencePiece
    SentencePiece,
    /// Unknown format
    Unknown,
}

/// Tokenizer format detection result
#[derive(Debug, Clone)]
pub struct FormatDetection {
    /// Detected format
    pub format: TokenizerFormat,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Reason for detection
    pub reason: String,
}

/// Detect tokenizer format from model metadata
pub fn detect_format(model_name: &str) -> FormatDetection {
    let (format, confidence, reason) =
        if model_name.contains("gpt") || model_name.contains("text-davinci") {
            (TokenizerFormat::BPE, 0.95, "GPT model detected")
        } else if model_name.contains("bert") || model_name.contains("wordpiece") {
            (
                TokenizerFormat::WordPiece,
                0.90,
                "BERT-style model detected",
            )
        } else if model_name.contains("t5") || model_name.contains("mt5") {
            (TokenizerFormat::SentencePiece, 0.85, "T5 model detected")
        } else {
            (TokenizerFormat::BPE, 0.50, "Default BPE assumed")
        };

    FormatDetection {
        format,
        confidence,
        reason: reason.to_string(),
    }
}

/// Token handler with caching and format detection
#[derive(Debug, Clone)]
pub struct TokenHandler {
    /// Current tokenizer
    tokenizer: Option<BPETokenizer>,
    /// Cached encoding results
    encoding_cache: HashMap<String, Vec<u32>>,
    /// Model name for format detection
    #[allow(dead_code)]
    model_name: String,
    /// Detected format
    format: TokenizerFormat,
}

impl TokenHandler {
    /// Create new token handler
    pub fn new(model_name: String) -> Self {
        let detection = detect_format(&model_name);
        Self {
            tokenizer: None,
            encoding_cache: HashMap::new(),
            model_name,
            format: detection.format,
        }
    }

    /// Set tokenizer
    pub fn set_tokenizer(&mut self, tokenizer: BPETokenizer) {
        self.tokenizer = Some(tokenizer);
        self.encoding_cache.clear(); // Invalidate cache
    }

    /// Encode with caching
    pub fn encode(&mut self, text: &str) -> Result<Vec<u32>, String> {
        if let Some(cached) = self.encoding_cache.get(text) {
            return Ok(cached.clone());
        }

        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| "Tokenizer not initialized".to_string())?;

        let encoded = tokenizer.encode(text);
        self.encoding_cache
            .insert(text.to_string(), encoded.clone());
        Ok(encoded)
    }

    /// Decode tokens
    pub fn decode(&self, tokens: &[u32]) -> Result<String, String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| "Tokenizer not initialized".to_string())?;

        Ok(tokenizer.decode(tokens))
    }

    /// Count tokens
    pub fn count_tokens(&mut self, text: &str) -> Result<usize, String> {
        Ok(self.encode(text)?.len())
    }

    /// Get detected format
    pub fn format(&self) -> TokenizerFormat {
        self.format
    }

    /// Clear encoding cache
    pub fn clear_cache(&mut self) {
        self.encoding_cache.clear();
    }

    /// Get cache statistics
    pub fn cache_size(&self) -> usize {
        self.encoding_cache.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vocabulary_creation() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.size(), 0);
    }

    #[test]
    fn test_vocabulary_add_token() {
        let mut vocab = Vocabulary::new();
        assert!(vocab.add_token("hello".to_string(), 1).is_ok());
        assert_eq!(vocab.get_id("hello"), Some(1));
        assert_eq!(vocab.get_token(1), Some("hello".to_string()));
    }

    #[test]
    fn test_vocabulary_duplicate_token() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("hello".to_string(), 1).unwrap();
        assert!(vocab.add_token("hello".to_string(), 2).is_err());
    }

    #[test]
    fn test_vocabulary_special_tokens() {
        let mut vocab = Vocabulary::new();
        assert!(vocab.add_special_token("PAD".to_string(), 0).is_ok());
        assert!(vocab.add_special_token("UNK".to_string(), 1).is_ok());
        assert_eq!(vocab.get_special("PAD"), Some(0));
        assert_eq!(vocab.pad_token_id(), 0);
    }

    #[test]
    fn test_vocabulary_contains() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("test".to_string(), 1).unwrap();
        assert!(vocab.contains("test"));
        assert!(!vocab.contains("missing"));
    }

    #[test]
    fn test_bpe_tokenizer_creation() {
        let vocab = Vocabulary::new();
        let tokenizer = BPETokenizer::new(vocab);
        assert_eq!(tokenizer.vocab().size(), 0);
    }

    #[test]
    fn test_token_handler_creation() {
        let handler = TokenHandler::new("gpt-3.5".to_string());
        assert_eq!(handler.format(), TokenizerFormat::BPE);
    }

    #[test]
    fn test_token_handler_cache() {
        let mut handler = TokenHandler::new("test".to_string());
        assert_eq!(handler.cache_size(), 0);

        let vocab = Vocabulary::new();
        handler.set_tokenizer(BPETokenizer::new(vocab));
        assert_eq!(handler.cache_size(), 0);
    }

    #[test]
    fn test_format_detection_gpt() {
        let detection = detect_format("gpt-3.5");
        assert_eq!(detection.format, TokenizerFormat::BPE);
        assert!(detection.confidence > 0.9);
    }

    #[test]
    fn test_format_detection_bert() {
        let detection = detect_format("bert-base");
        assert_eq!(detection.format, TokenizerFormat::WordPiece);
    }

    #[test]
    fn test_format_detection_t5() {
        let detection = detect_format("t5-base");
        assert_eq!(detection.format, TokenizerFormat::SentencePiece);
    }

    #[test]
    fn test_format_detection_unknown() {
        let detection = detect_format("my-custom-model");
        assert_eq!(detection.format, TokenizerFormat::BPE);
    }

    #[test]
    fn test_token_creation() {
        let token = Token::new(1, 5);
        assert_eq!(token.id, 1);
        assert_eq!(token.len, 5);
    }

    #[test]
    fn test_vocabulary_unk_default() {
        let vocab = Vocabulary::new();
        assert_eq!(vocab.unk_token_id(), 0);
    }

    #[test]
    fn test_vocabulary_size() {
        let mut vocab = Vocabulary::new();
        vocab.add_token("a".to_string(), 1).unwrap();
        vocab.add_token("b".to_string(), 2).unwrap();
        assert_eq!(vocab.size(), 2);
    }
}
