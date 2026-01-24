use super::circuit_breaker_config::CircuitBreakerConfig;
use super::circuit_breaker_transitions::CircuitState;
use super::circuit_state_transitions::StateTransitionHelper;

/// Handles state transitions when recording success/failure events
pub struct CircuitBreakerStateRecorder;

impl CircuitBreakerStateRecorder {
    /// Record a successful operation and transition state if needed
    pub fn record_success(
        state: CircuitState,
        transitions: &StateTransitionHelper,
        config: &CircuitBreakerConfig,
    ) {
        match state {
            CircuitState::Closed => {
                transitions.reset_failures();
            }
            CircuitState::HalfOpen => {
                let successes = transitions.increment_successes();
                if successes >= config.half_open_max_calls {
                    transitions.transition_to_closed();
                }
            }
            _ => {}
        }
    }

    /// Record a failed operation and transition state if needed
    pub fn record_failure(
        state: CircuitState,
        transitions: &StateTransitionHelper,
        config: &CircuitBreakerConfig,
    ) {
        match state {
            CircuitState::Closed => {
                let failures = transitions.increment_failures();
                if failures >= config.failure_threshold {
                    transitions.transition_to_open();
                }
            }
            CircuitState::HalfOpen => {
                transitions.transition_to_open();
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_success_resets_failures_in_closed_state() {
        let helper = StateTransitionHelper::new();
        helper.increment_failures();
        helper.increment_failures();
        let config = CircuitBreakerConfig::default();

        CircuitBreakerStateRecorder::record_success(CircuitState::Closed, &helper, &config);

        assert_eq!(helper.get_failures(), 0);
    }

    #[test]
    fn test_record_success_in_half_open_transitions_to_closed() {
        let helper = StateTransitionHelper::new();
        helper.transition_to_half_open();
        let config = CircuitBreakerConfig {
            half_open_max_calls: 1,
            ..Default::default()
        };

        CircuitBreakerStateRecorder::record_success(CircuitState::HalfOpen, &helper, &config);

        assert_eq!(helper.get_state(), 0); // Closed
    }

    #[test]
    fn test_record_failure_in_closed_transitions_to_open() {
        let helper = StateTransitionHelper::new();
        let config = CircuitBreakerConfig {
            failure_threshold: 1,
            ..Default::default()
        };

        CircuitBreakerStateRecorder::record_failure(CircuitState::Closed, &helper, &config);

        assert_eq!(helper.get_state(), 1); // Open
    }

    #[test]
    fn test_record_failure_in_half_open_transitions_to_open() {
        let helper = StateTransitionHelper::new();
        helper.transition_to_half_open();
        let config = CircuitBreakerConfig::default();

        CircuitBreakerStateRecorder::record_failure(CircuitState::HalfOpen, &helper, &config);

        assert_eq!(helper.get_state(), 1); // Open
    }
}
