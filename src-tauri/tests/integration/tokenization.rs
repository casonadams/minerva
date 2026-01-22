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
    assert_eq!(tokenizer.merge_count(), 0);
    let tokens = tokenizer.encode("abc");
    assert!(tokens.is_empty()); // Empty vocab, so no tokens
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
    assert!(!handler.is_initialized());
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
    assert!(handler.is_initialized());
}

#[test]
fn test_token_handler_is_initialized() {
    let mut handler = TokenHandler::new("test".to_string());
    assert!(!handler.is_initialized());

    let vocab = Vocabulary::new();
    handler.set_tokenizer(BPETokenizer::new(vocab));
    assert!(handler.is_initialized());
}

#[test]
fn test_token_handler_encode_without_tokenizer() {
    let handler = TokenHandler::new("test".to_string());
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
    let handler = TokenHandler::new("test".to_string());
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

// Vocabulary Loading Tests

#[test]
fn test_load_vocabulary_txt_basic() {
    use minerva_lib::inference::tokenizer::load_vocabulary_txt;

    let txt = "hello 1\nworld 2\ntest 3\n";
    let vocab = load_vocabulary_txt(txt).unwrap();

    assert_eq!(vocab.get_id("hello"), Some(1));
    assert_eq!(vocab.get_id("world"), Some(2));
    assert_eq!(vocab.get_id("test"), Some(3));
}

#[test]
fn test_load_vocabulary_txt_special_tokens() {
    use minerva_lib::inference::tokenizer::load_vocabulary_txt;

    let txt = "PAD 0\nUNK 1\nhello 2\n";
    let vocab = load_vocabulary_txt(txt).unwrap();

    assert_eq!(vocab.pad_token_id(), 0);
    assert_eq!(vocab.unk_token_id(), 1);
}

#[test]
fn test_load_vocabulary_txt_comments() {
    use minerva_lib::inference::tokenizer::load_vocabulary_txt;

    let txt = "# This is a comment\nhello 1\n# Another comment\nworld 2\n";
    let vocab = load_vocabulary_txt(txt).unwrap();

    assert_eq!(vocab.size(), 2);
}

#[test]
fn test_load_vocabulary_txt_empty_lines() {
    use minerva_lib::inference::tokenizer::load_vocabulary_txt;

    let txt = "hello 1\n\nworld 2\n\n";
    let vocab = load_vocabulary_txt(txt).unwrap();

    assert_eq!(vocab.size(), 2);
}

#[test]
fn test_load_vocabulary_txt_invalid() {
    use minerva_lib::inference::tokenizer::load_vocabulary_txt;

    let txt = "# Only comments\n";
    let result = load_vocabulary_txt(txt);

    assert!(result.is_err());
}

#[test]
fn test_load_vocabulary_json_basic() {
    use minerva_lib::inference::tokenizer::load_vocabulary_json;

    let json = r#"{"tokens": {"hello": 1, "world": 2}}"#;
    let vocab = load_vocabulary_json(json).unwrap();

    assert!(vocab.size() > 0);
}

// BPE Merge Tests

#[test]
fn test_bpe_add_merge_valid() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("h".to_string(), 1).unwrap();
    vocab.add_token("e".to_string(), 2).unwrap();
    vocab.add_token("he".to_string(), 3).unwrap();

    let mut tokenizer = BPETokenizer::new(vocab);
    assert!(tokenizer.add_merge(1, 2, 3).is_ok());
    assert_eq!(tokenizer.merge_count(), 1);
}

#[test]
fn test_bpe_add_merge_invalid_left() {
    let vocab = Vocabulary::new();
    let mut tokenizer = BPETokenizer::new(vocab);

    let result = tokenizer.add_merge(999, 2, 3);
    assert!(result.is_err());
}

#[test]
fn test_bpe_add_merge_invalid_right() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("h".to_string(), 1).unwrap();

    let mut tokenizer = BPETokenizer::new(vocab);
    let result = tokenizer.add_merge(1, 999, 3);
    assert!(result.is_err());
}

#[test]
fn test_bpe_merge_count() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("a".to_string(), 1).unwrap();
    vocab.add_token("b".to_string(), 2).unwrap();
    vocab.add_token("c".to_string(), 3).unwrap();
    vocab.add_token("ab".to_string(), 4).unwrap();
    vocab.add_token("abc".to_string(), 5).unwrap();

    let mut tokenizer = BPETokenizer::new(vocab);
    assert_eq!(tokenizer.merge_count(), 0);

    tokenizer.add_merge(1, 2, 4).ok();
    assert_eq!(tokenizer.merge_count(), 1);

    tokenizer.add_merge(4, 3, 5).ok();
    assert_eq!(tokenizer.merge_count(), 2);
}

// Real BPE Pipeline Tests

#[test]
fn test_real_bpe_pipeline() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("h".to_string(), 1).unwrap();
    vocab.add_token("e".to_string(), 2).unwrap();
    vocab.add_token("l".to_string(), 3).unwrap();
    vocab.add_token("o".to_string(), 4).unwrap();

    let mut tokenizer = BPETokenizer::new(vocab);
    tokenizer.add_merge(1, 2, 5).ok();

    let tokens = tokenizer.encode("hello");
    assert!(!tokens.is_empty());
}

#[test]
fn test_vocabulary_with_tokenizer_and_merges() {
    let mut vocab = Vocabulary::new();
    vocab.add_token("a".to_string(), 1).unwrap();
    vocab.add_token("b".to_string(), 2).unwrap();
    vocab.add_token("c".to_string(), 3).unwrap();

    let mut tokenizer = BPETokenizer::new(vocab);
    tokenizer.add_merge(1, 2, 4).ok();

    assert_eq!(tokenizer.merge_count(), 1);
    let tokens = tokenizer.encode("abc");
    assert!(!tokens.is_empty());
}
