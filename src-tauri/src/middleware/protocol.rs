//! Protocol Compliance Middleware
//! Ensures all API requests and responses follow OpenAI-compatible standards

use axum::body::Body;
use axum::http::Request;
use axum::middleware::Next;
use axum::response::Response;

/// Request ID from X-Request-ID header
#[derive(Clone)]
pub struct RequestId(pub String);

/// Protocol version header
pub const API_VERSION: &str = "0.1.0";
pub const OPENAI_COMPATIBLE_VERSION: &str = "OpenAI compatible";

/// Middleware to add protocol headers to responses
pub async fn add_protocol_headers(mut req: Request<Body>, next: Next) -> Response {
    // Generate request ID if not present
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("")
        .to_string();

    let request_id = if request_id.is_empty() {
        uuid::Uuid::new_v4().to_string()
    } else {
        request_id
    };

    // Store request ID in extensions for handlers
    req.extensions_mut().insert(RequestId(request_id.clone()));

    let mut response = next.run(req).await;

    // Add protocol headers
    response.headers_mut().insert(
        "X-Request-ID",
        request_id
            .parse()
            .unwrap_or_else(|_| "unknown".parse().unwrap()),
    );
    response
        .headers_mut()
        .insert("X-API-Version", API_VERSION.parse().unwrap());

    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_version_constant() {
        assert_eq!(API_VERSION, "0.1.0", "API version should be set");
    }

    #[test]
    fn test_openai_compatible_version() {
        assert!(
            !OPENAI_COMPATIBLE_VERSION.is_empty(),
            "OpenAI version should be set"
        );
    }
}
