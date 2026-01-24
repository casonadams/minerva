/// System resource state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceState {
    /// Plenty of resources available
    Abundant,
    /// Moderate resource usage
    Moderate,
    /// High resource usage, approaching limits
    Constrained,
    /// Critical resource shortage
    Critical,
}

impl ResourceState {
    /// Classify based on available memory percentage
    pub fn from_memory_percent(available_percent: f64) -> Self {
        match available_percent {
            p if p > 50.0 => ResourceState::Abundant,
            p if p > 30.0 => ResourceState::Moderate,
            p if p > 10.0 => ResourceState::Constrained,
            _ => ResourceState::Critical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abundant() {
        assert_eq!(
            ResourceState::from_memory_percent(80.0),
            ResourceState::Abundant
        );
    }

    #[test]
    fn test_moderate() {
        assert_eq!(
            ResourceState::from_memory_percent(40.0),
            ResourceState::Moderate
        );
    }

    #[test]
    fn test_constrained() {
        assert_eq!(
            ResourceState::from_memory_percent(15.0),
            ResourceState::Constrained
        );
    }

    #[test]
    fn test_critical() {
        assert_eq!(
            ResourceState::from_memory_percent(5.0),
            ResourceState::Critical
        );
    }
}
