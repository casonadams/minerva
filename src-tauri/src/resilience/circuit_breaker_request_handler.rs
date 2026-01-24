use super::circuit_breaker_config::CircuitBreakerConfig;
use super::circuit_breaker_transitions::CircuitState;
use super::circuit_state_transitions::StateTransitionHelper;
use std::time::Duration;

/// Handles request validation logic for circuit breaker states
pub struct CircuitBreakerRequestHandler;

impl CircuitBreakerRequestHandler {
    /// Check if request should be allowed based on current state
    pub fn should_allow_request(
        state: CircuitState,
        transitions: &StateTransitionHelper,
        config: &CircuitBreakerConfig,
    ) -> bool {
        match state {
            CircuitState::Closed => true,
            CircuitState::Open => Self::handle_open_state(transitions, config),
            CircuitState::HalfOpen => Self::handle_half_open_state(transitions, config),
        }
    }

    /// Handle request when circuit is open
    fn handle_open_state(
        transitions: &StateTransitionHelper,
        config: &CircuitBreakerConfig,
    ) -> bool {
        match transitions.time_since_open() {
            Ok(elapsed) if elapsed >= Duration::from_secs(config.timeout_secs) => {
                transitions.transition_to_half_open();
                true
            }
            _ => false,
        }
    }

    /// Handle request when circuit is half-open
    fn handle_half_open_state(
        transitions: &StateTransitionHelper,
        config: &CircuitBreakerConfig,
    ) -> bool {
        let successes = transitions.get_successes();
        successes < config.half_open_max_calls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allow_request_when_closed() {
        let helper = StateTransitionHelper::new();
        let config = CircuitBreakerConfig::default();

        assert!(CircuitBreakerRequestHandler::should_allow_request(
            CircuitState::Closed,
            &helper,
            &config,
        ));
    }

    #[test]
    fn test_deny_request_when_open_and_timeout_not_elapsed() {
        let helper = StateTransitionHelper::new();
        helper.transition_to_open();
        let config = CircuitBreakerConfig {
            timeout_secs: 100,
            ..Default::default()
        };

        assert!(!CircuitBreakerRequestHandler::should_allow_request(
            CircuitState::Open,
            &helper,
            &config,
        ));
    }

    #[test]
    fn test_deny_request_when_half_open_and_max_calls_reached() {
        let helper = StateTransitionHelper::new();
        helper.transition_to_half_open();
        helper.increment_successes();
        let config = CircuitBreakerConfig {
            half_open_max_calls: 1,
            ..Default::default()
        };

        assert!(!CircuitBreakerRequestHandler::should_allow_request(
            CircuitState::HalfOpen,
            &helper,
            &config,
        ));
    }

    #[test]
    fn test_allow_request_when_half_open_and_calls_available() {
        let helper = StateTransitionHelper::new();
        helper.transition_to_half_open();
        let config = CircuitBreakerConfig {
            half_open_max_calls: 3,
            ..Default::default()
        };

        assert!(CircuitBreakerRequestHandler::should_allow_request(
            CircuitState::HalfOpen,
            &helper,
            &config,
        ));
    }
}
