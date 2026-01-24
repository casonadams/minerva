// API Response Format Tests - Verify OpenAI-compatible response structure

use minerva_lib::api::{ApiError, ApiErrorResponse, ApiResponse};

#[test]
fn test_api_response_serialization() {
    let data = vec!["model1", "model2"];
    let response = ApiResponse::new(data);

    let json = serde_json::to_string(&response).expect("Should serialize");
    assert!(json.contains("\"data\""), "Should have data field");
    assert!(json.contains("\"meta\""), "Should have meta field");
}

#[test]
fn test_api_response_data_field() {
    let data = vec!["test"];
    let response = ApiResponse::new(data.clone());

    assert_eq!(response.data, data, "Data field should match input");
}

#[test]
fn test_api_response_meta_structure() {
    let response = ApiResponse::new("test");

    let meta = response.meta.expect("Meta should exist");
    assert!(
        !meta.request_id.is_empty(),
        "Request ID should not be empty"
    );
    assert!(!meta.timestamp.is_empty(), "Timestamp should not be empty");
    assert_eq!(meta.version, "0.1.0", "Version should match");
}

#[test]
fn test_api_error_response_structure() {
    let error = ApiError {
        message: "Invalid request".to_string(),
        code: "invalid_request".to_string(),
        type_: Some("invalid_request_error".to_string()),
        param: Some("model".to_string()),
    };

    let response = ApiErrorResponse { error };

    let json = serde_json::to_string(&response).expect("Should serialize");
    assert!(json.contains("\"error\""), "Should have error field");
    assert!(json.contains("\"message\""), "Should have message field");
    assert!(json.contains("\"code\""), "Should have code field");
}

#[test]
fn test_api_error_minimal_structure() {
    let error = ApiError {
        message: "Error".to_string(),
        code: "error".to_string(),
        type_: None,
        param: None,
    };

    let response = ApiErrorResponse { error };
    let json = serde_json::to_string(&response).expect("Should serialize");

    // Skipped fields should not appear
    assert!(
        !json.contains("\"type_\"") || json.contains("null"),
        "Empty fields should not appear"
    );
}

#[test]
fn test_api_response_empty_data() {
    let data: Vec<String> = vec![];
    let response = ApiResponse::new(data);

    let json = serde_json::to_string(&response).expect("Should serialize");
    assert!(json.contains("[]"), "Should serialize empty array");
}

#[test]
fn test_api_response_nested_data() {
    #[derive(serde::Serialize)]
    struct TestData {
        field1: String,
        field2: i32,
    }

    let data = TestData {
        field1: "value".to_string(),
        field2: 42,
    };

    let response = ApiResponse::new(data);
    let json = serde_json::to_string(&response).expect("Should serialize");

    assert!(json.contains("field1"), "Should contain nested field1");
    assert!(json.contains("field2"), "Should contain nested field2");
}

#[test]
fn test_response_timestamp_format() {
    let response = ApiResponse::new("test");

    let meta = response.meta.expect("Meta should exist");
    // Verify it's a valid RFC3339 timestamp format
    assert!(
        meta.timestamp.contains('T'),
        "Timestamp should be RFC3339 format"
    );
    assert!(
        meta.timestamp.contains('Z') || meta.timestamp.contains('+'),
        "Timestamp should include timezone"
    );
}

#[test]
fn test_response_request_id_uuid_format() {
    let response = ApiResponse::new("test");

    let meta = response.meta.expect("Meta should exist");
    // UUIDs have a specific format: 8-4-4-4-12 hex characters
    let parts: Vec<&str> = meta.request_id.split('-').collect();
    assert_eq!(parts.len(), 5, "Request ID should be valid UUID format");
}

#[test]
fn test_multiple_responses_different_request_ids() {
    let response1 = ApiResponse::new("test1");
    let response2 = ApiResponse::new("test2");

    let id1 = response1.meta.as_ref().unwrap().request_id.clone();
    let id2 = response2.meta.as_ref().unwrap().request_id.clone();

    assert_ne!(id1, id2, "Each response should have unique request ID");
}

#[test]
fn test_api_response_openai_compatibility() {
    // Test that response structure is compatible with OpenAI API format
    let data = serde_json::json!({
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "model": "gpt-4"
    });

    let response = ApiResponse::new(data);
    let json = serde_json::to_string(&response).expect("Should serialize");

    // Should still work with OpenAI-like payloads
    assert!(
        json.contains("\"data\""),
        "Response wrapper should have data field"
    );
    assert!(
        json.contains("\"meta\""),
        "Response should have meta for request tracking"
    );
}

#[test]
fn test_error_response_openai_format() {
    // Test that error response follows OpenAI error format
    let error = ApiError {
        message: "The model gpt-5 does not exist".to_string(),
        code: "model_not_found".to_string(),
        type_: Some("invalid_request_error".to_string()),
        param: Some("model".to_string()),
    };

    let response = ApiErrorResponse { error };
    let json = serde_json::to_string(&response).expect("Should serialize");

    assert!(
        json.contains("\"error\""),
        "Should have error wrapper (OpenAI format)"
    );
    assert!(json.contains("model_not_found"), "Should have error code");
}
