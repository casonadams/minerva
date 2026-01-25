/// Circuit Breaker Pattern
///
/// Protects against cascading failures by:
/// - Closed: Normal operation, requests pass through
/// - Open: Too many failures, requests fail immediately
/// - Half-Open: Testing if service recovered, single request allowed
///
/// Transitions:
/// - Closed → Open: Failure threshold exceeded
/// - Open → Half-Open: Timeout elapsed
/// - Half-Open → Closed: Test request succeeds
/// - Half-Open → Open: Test request fails
pub use super::circuit_breaker_config::CircuitBreakerConfig;
pub use super::circuit_breaker_facade::CircuitBreaker;
pub use super::circuit_breaker_transitions::CircuitState;
