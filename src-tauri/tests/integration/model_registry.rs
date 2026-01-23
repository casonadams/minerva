//! Model registry integration tests
//!
//! Tests for model registry creation, cache management, and model discovery.
//! Covers registry lifecycle, cache limits, and model lookup operations.

use minerva_lib::inference::model_registry::ModelRegistry;

#[test]
fn test_model_registry_creation() {
    let registry = ModelRegistry::new();
    assert!(registry.list().is_empty());
    assert_eq!(registry.cached_size_mb(), 0);
}

#[test]
fn test_model_registry_default() {
    let registry = ModelRegistry::default();
    assert_eq!(registry.list().len(), 0);
    assert_eq!(registry.list_cached().len(), 0);
}

#[test]
fn test_model_registry_cache_usage() {
    let registry = ModelRegistry::new();
    assert_eq!(registry.cache_usage_percent(), 0.0);
}

#[test]
fn test_model_registry_max_cache_size() {
    let mut registry = ModelRegistry::new();
    registry.set_max_cache_size(10000);

    assert!(!registry.would_exceed_limit(5000)); // Within limit
    assert!(registry.would_exceed_limit(10001)); // Exceeds limit
    assert!(!registry.would_exceed_limit(9999)); // Within limit
}

#[test]
fn test_model_registry_oldest_cached() {
    let registry = ModelRegistry::new();
    assert!(registry.oldest_cached().is_empty());
}

#[test]
fn test_model_registry_least_used() {
    let registry = ModelRegistry::new();
    assert!(registry.least_used_cached().is_empty());
}

#[test]
fn test_model_registry_remove() {
    let mut registry = ModelRegistry::new();
    assert!(registry.remove("nonexistent").is_none());
}

#[test]
fn test_model_registry_clear() {
    let mut registry = ModelRegistry::new();
    registry.clear();
    assert!(registry.list().is_empty());
}

#[test]
fn test_model_registry_get_nonexistent() {
    let registry = ModelRegistry::new();
    assert!(registry.get("nonexistent").is_none());
}
