// Real Tokenization and Vocabulary Integration Tests (Phase 4 Step 4)

use minerva_lib::inference::tokenizer::{
    BPETokenizer, FormatDetection, Token, TokenHandler, TokenizerFormat, Vocabulary,
};

// Vocabulary Tests

#[test]
fn test_vocabulary_initialization() {
    let vocab = Vocabulary::new();
    assert_eq!(vocab.size(), 0);
    assert!(!vocab.contains("test"));
}

#[test]
fn test_vocabulary_add_single_token() {
    let mut vocab = Vocabulary::new();
    assert!(vocab.add_token("hello".to_string(), 1).is_ok());
    assert_eq!(vocab.get_id("hello"), Some(1));
    assert_eq!(vocab.get_token(1), Some("hello".to_string()));
}

#[test]
fn test_vocabulary_add_multiple_tokens() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("hello".to_string(), 1).unwrap();
    vocab.add_token("world".to_string(), 2).unwrap();
    vocab.add_token("test".to_string(), 3).unwrap();

    assert_eq!(vocab.size(), 3);
    assert!(vocab.contains("hello"));
    assert!(vocab.contains("world"));
    assert!(vocab.contains("test"));
}

#[test]
fn test_vocabulary_duplicate_rejection() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("hello".to_string(), 1).unwrap();
    assert!(vocab.add_token("hello".to_string(), 2).is_err());
}

#[test]
fn test_vocabulary_bidirectional_lookup() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("token".to_string(), 42).unwrap();

    assert_eq!(vocab.get_id("token"), Some(42));
    assert_eq!(vocab.get_token(42), Some("token".to_string()));
    assert_eq!(vocab.get_id("missing"), None);
    assert_eq!(vocab.get_token(999), None);
}

#[test]
fn test_vocabulary_special_tokens() {
    let mut vocab = Vocabulary::new();
    vocab.add_special_token("PAD".to_string(), 0).unwrap();
    vocab.add_special_token("UNK".to_string(), 1).unwrap();
    vocab.add_special_token("EOS".to_string(), 2).unwrap();
    vocab.add_special_token("BOS".to_string(), 3).unwrap();

    assert_eq!(vocab.get_special("PAD"), Some(0));
    assert_eq!(vocab.get_special("UNK"), Some(1));
    assert_eq!(vocab.pad_token_id(), 0);
    assert_eq!(vocab.unk_token_id(), 1);
}

#[test]
fn test_vocabulary_special_token_duplicate() {
    let mut vocab = Vocabulary::new();
    vocab.add_special_token("PAD".to_string(), 0).unwrap();
    assert!(vocab.add_special_token("PAD".to_string(), 1).is_err());
}

#[test]
fn test_vocabulary_default_special_tokens() {
    let vocab = Vocabulary::new();
    assert_eq!(vocab.pad_token_id(), 0);
    assert_eq!(vocab.unk_token_id(), 0);
}

// Token Structure Tests

#[test]
fn test_token_creation() {
    let token = Token::new(1, 5);
    assert_eq!(token.id, 1);
    assert_eq!(token.len, 5);
}

#[test]
fn test_token_equality() {
    let token1 = Token::new(1, 5);
    let token2 = Token::new(1, 5);
    let token3 = Token::new(2, 5);

    assert_eq!(token1, token2);
    assert_ne!(token1, token3);
}

// BPE Tokenizer Tests

#[test]
fn test_bpe_tokenizer_creation() {
    let vocab = Vocabulary::new();
    let tokenizer = BPETokenizer::new(vocab);
    assert_eq!(tokenizer.vocab().size(), 0);
}

#[test]
fn test_bpe_tokenizer_with_vocab() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("a".to_string(), 1).unwrap();
    vocab.add_token("b".to_string(), 2).unwrap();

    let tokenizer = BPETokenizer::new(vocab);
    assert_eq!(tokenizer.vocab().size(), 2);
}

#[test]
fn test_bpe_encode_empty_text() {
    let vocab = Vocabulary::new();
    let tokenizer = BPETokenizer::new(vocab);

    let tokens = tokenizer.encode("");
    assert!(tokens.is_empty());
}

#[test]
fn test_bpe_decode_empty_tokens() {
    let vocab = Vocabulary::new();
    let tokenizer = BPETokenizer::new(vocab);

    let text = tokenizer.decode(&[]);
    assert_eq!(text, "");
}

#[test]
fn test_bpe_count_tokens() {
    let mut vocab = Vocabulary::new();
    // Add some basic tokens
    for (i, c) in ['a', 'b', 'c', ' '].iter().enumerate() {
        let char_str = c.to_string();
        vocab.add_token(char_str, (i + 1) as u32).unwrap();
    }

    let tokenizer = BPETokenizer::new(vocab);
    let count = tokenizer.count_tokens("abc");
    assert!(count > 0);
}

// Format Detection Tests

#[test]
fn test_format_detection_gpt_model() {
    let detection = FormatDetection {
        format: TokenizerFormat::BPE,
        confidence: 0.95,
        reason: "GPT model".to_string(),
    };

    assert_eq!(detection.format, TokenizerFormat::BPE);
    assert!(detection.confidence > 0.9);
}

#[test]
fn test_detect_gpt_format() {
    use minerva_lib::inference::tokenizer::detect_format;

    let detection = detect_format("gpt-3.5");
    assert_eq!(detection.format, TokenizerFormat::BPE);
    assert!(detection.confidence > 0.9);
}

#[test]
fn test_detect_bert_format() {
    use minerva_lib::inference::tokenizer::detect_format;

    let detection = detect_format("bert-base-uncased");
    assert_eq!(detection.format, TokenizerFormat::WordPiece);
}

#[test]
fn test_detect_t5_format() {
    use minerva_lib::inference::tokenizer::detect_format;

    let detection = detect_format("t5-small");
    assert_eq!(detection.format, TokenizerFormat::SentencePiece);
}

#[test]
fn test_detect_default_format() {
    use minerva_lib::inference::tokenizer::detect_format;

    let detection = detect_format("custom-model-v1");
    assert_eq!(detection.format, TokenizerFormat::BPE);
}

// Token Handler Tests

#[test]
fn test_token_handler_creation() {
    let handler = TokenHandler::new("gpt-3.5".to_string());
    assert_eq!(handler.format(), TokenizerFormat::BPE);
    assert_eq!(handler.cache_size(), 0);
}

#[test]
fn test_token_handler_format_detection() {
    let handler_gpt = TokenHandler::new("gpt-3.5".to_string());
    let handler_bert = TokenHandler::new("bert-base".to_string());
    let handler_t5 = TokenHandler::new("t5-small".to_string());

    assert_eq!(handler_gpt.format(), TokenizerFormat::BPE);
    assert_eq!(handler_bert.format(), TokenizerFormat::WordPiece);
    assert_eq!(handler_t5.format(), TokenizerFormat::SentencePiece);
}

#[test]
fn test_token_handler_set_tokenizer() {
    let mut handler = TokenHandler::new("test".to_string());
    let vocab = Vocabulary::new();
    let tokenizer = BPETokenizer::new(vocab);

    handler.set_tokenizer(tokenizer);
    // Cache should be cleared when tokenizer is set
    assert_eq!(handler.cache_size(), 0);
}

#[test]
fn test_token_handler_clear_cache() {
    let mut handler = TokenHandler::new("test".to_string());
    let vocab = Vocabulary::new();
    handler.set_tokenizer(BPETokenizer::new(vocab));

    handler.clear_cache();
    assert_eq!(handler.cache_size(), 0);
}

#[test]
fn test_token_handler_encode_without_tokenizer() {
    let mut handler = TokenHandler::new("test".to_string());
    let result = handler.encode("hello");

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not initialized"));
}

#[test]
fn test_token_handler_decode_without_tokenizer() {
    let handler = TokenHandler::new("test".to_string());
    let result = handler.decode(&[1, 2, 3]);

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not initialized"));
}

#[test]
fn test_token_handler_count_without_tokenizer() {
    let mut handler = TokenHandler::new("test".to_string());
    let result = handler.count_tokens("hello");

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not initialized"));
}

// Integration Tests

#[test]
fn test_vocabulary_and_tokenizer_integration() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("hello".to_string(), 1).unwrap();
    vocab.add_token("world".to_string(), 2).unwrap();

    let tokenizer = BPETokenizer::new(vocab);
    assert_eq!(tokenizer.vocab().size(), 2);
}

#[test]
fn test_full_tokenization_pipeline() {
    let mut handler = TokenHandler::new("gpt-3.5".to_string());
    let mut vocab = Vocabulary::new();
    vocab.add_special_token("PAD".to_string(), 0).unwrap();
    vocab.add_token("test".to_string(), 1).unwrap();

    let tokenizer = BPETokenizer::new(vocab);
    handler.set_tokenizer(tokenizer);

    assert_eq!(handler.format(), TokenizerFormat::BPE);
}

#[test]
fn test_tokenizer_format_and_handler() {
    use minerva_lib::inference::tokenizer::detect_format;

    let detection = detect_format("gpt-davinci");
    assert_eq!(detection.format, TokenizerFormat::BPE);

    let handler = TokenHandler::new("gpt-davinci".to_string());
    assert_eq!(handler.format(), detection.format);
}
