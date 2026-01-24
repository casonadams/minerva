// API Protocol Unification Tests - OpenAI compatibility and response standardization

use minerva_lib::api::{ApiError, ApiResponse, ProtocolValidator};

#[test]
fn test_protocol_validator_model_id_valid() {
    let result = ProtocolValidator::validate_model_id("gpt-4");
    assert!(result.is_ok(), "Valid model ID should pass");
}

#[test]
fn test_protocol_validator_model_id_empty() {
    let result = ProtocolValidator::validate_model_id("");

    assert!(result.is_err(), "Empty model ID should fail");
    let err = result.unwrap_err();
    assert_eq!(
        err.code, "invalid_model_id",
        "Error code should be invalid_model_id"
    );
}

#[test]
fn test_protocol_validator_model_id_various_formats() {
    assert!(ProtocolValidator::validate_model_id("gpt-4").is_ok());
    assert!(ProtocolValidator::validate_model_id("gpt-4-turbo").is_ok());
    assert!(ProtocolValidator::validate_model_id("claude-3").is_ok());
    assert!(ProtocolValidator::validate_model_id("llama-2-7b").is_ok());
}

#[test]
fn test_protocol_validator_model_id_max_length() {
    let valid_id = "a".repeat(256);
    assert!(
        ProtocolValidator::validate_model_id(&valid_id).is_ok(),
        "256 char ID should pass"
    );

    let invalid_id = "a".repeat(257);
    assert!(
        ProtocolValidator::validate_model_id(&invalid_id).is_err(),
        "257 char ID should fail"
    );
}

#[test]
fn test_protocol_validator_temperature_boundaries() {
    assert!(ProtocolValidator::validate_temperature(0.0).is_ok());
    assert!(ProtocolValidator::validate_temperature(1.0).is_ok());
    assert!(ProtocolValidator::validate_temperature(2.0).is_ok());
    assert!(ProtocolValidator::validate_temperature(0.5).is_ok());
}

#[test]
fn test_protocol_validator_temperature_invalid() {
    assert!(ProtocolValidator::validate_temperature(-0.1).is_err());
    assert!(ProtocolValidator::validate_temperature(2.1).is_err());
    assert!(ProtocolValidator::validate_temperature(-1.0).is_err());
}

#[test]
fn test_protocol_validator_temperature_error_details() {
    let result = ProtocolValidator::validate_temperature(3.0);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code, "invalid_parameter");
    assert_eq!(err.param, Some("temperature".to_string()));
}

#[test]
fn test_protocol_validator_max_tokens_valid() {
    assert!(ProtocolValidator::validate_max_tokens(1).is_ok());
    assert!(ProtocolValidator::validate_max_tokens(256).is_ok());
    assert!(ProtocolValidator::validate_max_tokens(4096).is_ok());
}

#[test]
fn test_protocol_validator_max_tokens_boundaries() {
    // Zero should fail
    assert!(ProtocolValidator::validate_max_tokens(0).is_err());

    // Above max should fail
    assert!(ProtocolValidator::validate_max_tokens(4097).is_err());
    assert!(ProtocolValidator::validate_max_tokens(10000).is_err());
}

#[test]
fn test_protocol_validator_max_tokens_error_details() {
    let result = ProtocolValidator::validate_max_tokens(5000);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code, "invalid_parameter");
    assert_eq!(err.param, Some("max_tokens".to_string()));
}

#[test]
fn test_protocol_validator_top_p_valid() {
    assert!(ProtocolValidator::validate_top_p(0.0).is_ok());
    assert!(ProtocolValidator::validate_top_p(0.5).is_ok());
    assert!(ProtocolValidator::validate_top_p(0.9).is_ok());
    assert!(ProtocolValidator::validate_top_p(1.0).is_ok());
}

#[test]
fn test_protocol_validator_top_p_invalid() {
    assert!(ProtocolValidator::validate_top_p(-0.1).is_err());
    assert!(ProtocolValidator::validate_top_p(1.1).is_err());
    assert!(ProtocolValidator::validate_top_p(2.0).is_err());
}

#[test]
fn test_protocol_validator_top_p_error_details() {
    let result = ProtocolValidator::validate_top_p(1.5);

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert_eq!(err.code, "invalid_parameter");
    assert_eq!(err.param, Some("top_p".to_string()));
}

#[test]
fn test_api_response_creation() {
    let data = vec!["model1", "model2"];
    let response = ApiResponse::new(data.clone());

    assert_eq!(response.data, data);
    assert!(response.meta.is_some(), "Metadata should be present");
}

#[test]
fn test_api_response_metadata_fields() {
    let data = "test";
    let response = ApiResponse::new(data);

    let meta = response.meta.expect("Meta should exist");
    assert!(!meta.request_id.is_empty(), "Request ID should be set");
    assert!(!meta.timestamp.is_empty(), "Timestamp should be set");
    assert_eq!(meta.version, "0.1.0");
}

#[test]
fn test_api_response_without_metadata() {
    let data = vec!["test"];
    let response = ApiResponse::<Vec<&str>>::without_meta(data);

    assert!(response.meta.is_none(), "Meta should not be present");
}

#[test]
fn test_api_error_structure() {
    let error = ApiError {
        message: "Invalid request".to_string(),
        code: "invalid_request".to_string(),
        type_: Some("invalid_request_error".to_string()),
        param: Some("model".to_string()),
    };

    assert_eq!(error.message, "Invalid request");
    assert_eq!(error.code, "invalid_request");
    assert_eq!(error.type_, Some("invalid_request_error".to_string()));
    assert_eq!(error.param, Some("model".to_string()));
}

#[test]
fn test_api_error_minimal() {
    let error = ApiError {
        message: "Error occurred".to_string(),
        code: "error".to_string(),
        type_: None,
        param: None,
    };

    assert_eq!(error.message, "Error occurred");
    assert!(error.type_.is_none());
    assert!(error.param.is_none());
}

#[test]
fn test_protocol_error_codes() {
    // Test various error code patterns
    let codes = vec![
        "invalid_model_id",
        "invalid_parameter",
        "model_not_found",
        "rate_limit_exceeded",
        "server_error",
    ];

    for code in codes {
        let error = ApiError {
            message: "Test".to_string(),
            code: code.to_string(),
            type_: Some("test_error".to_string()),
            param: None,
        };

        // Error should have proper code
        assert_eq!(error.code, code);
    }
}

#[test]
fn test_validation_chain() {
    // Test multiple validations in sequence
    assert!(ProtocolValidator::validate_model_id("test-model").is_ok());
    assert!(ProtocolValidator::validate_temperature(1.5).is_ok());
    assert!(ProtocolValidator::validate_max_tokens(256).is_ok());
    assert!(ProtocolValidator::validate_top_p(0.9).is_ok());
}

#[test]
fn test_validation_failure_isolation() {
    // Failing one validation shouldn't affect others
    assert!(ProtocolValidator::validate_model_id("").is_err());
    assert!(ProtocolValidator::validate_temperature(1.5).is_ok());

    assert!(ProtocolValidator::validate_temperature(3.0).is_err());
    assert!(ProtocolValidator::validate_model_id("valid-model").is_ok());
}
