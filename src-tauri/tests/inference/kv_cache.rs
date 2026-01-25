use minerva::inference::kv_cache::{KVCache, KVCacheConfig, KVStoreParams};

#[test]
fn test_kv_cache_creation() {
    let config = KVCacheConfig {
        num_layers: 4,
        max_seq_len: 512,
        num_heads: 8,
        head_dim: 64,
    };
    let cache = KVCache::new(config);
    assert_eq!(cache.keys.len(), 4);
    assert_eq!(cache.keys[0].len(), 512);
}

#[test]
fn test_kv_cache_store_and_get() {
    let config = KVCacheConfig {
        num_layers: 1,
        max_seq_len: 10,
        num_heads: 2,
        head_dim: 4,
    };
    let mut cache = KVCache::new(config);
    let k = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let v = vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

    let params = KVStoreParams::builder(k, v).layer(0).pos(0).build();
    assert!(cache.store(params).is_ok());
    let (k_retrieved, v_retrieved) = cache.get(0, 0).unwrap();
    assert_eq!(k_retrieved.len(), 8);
    assert_eq!(v_retrieved.len(), 8);
}

#[test]
fn test_kv_cache_clear() {
    let config = KVCacheConfig {
        num_layers: 2,
        max_seq_len: 100,
        num_heads: 4,
        head_dim: 32,
    };
    let mut cache = KVCache::new(config);
    let k = vec![0.5; 128];
    let v = vec![0.5; 128];
    let params = KVStoreParams::builder(k, v).layer(0).pos(0).build();
    cache.store(params).unwrap();

    cache.clear();
    let (k_cleared, v_cleared) = cache.get(0, 0).unwrap();
    assert!(k_cleared.iter().all(|&v| v == 0.0));
    assert!(v_cleared.iter().all(|&v| v == 0.0));
}

#[test]
fn test_kv_cache_layer_out_of_bounds() {
    let config = KVCacheConfig {
        num_layers: 2,
        max_seq_len: 10,
        num_heads: 4,
        head_dim: 32,
    };
    let mut cache = KVCache::new(config);
    let k = vec![0.5; 128];
    let v = vec![0.5; 128];
    let params = KVStoreParams::builder(k, v).layer(5).pos(0).build();
    assert!(cache.store(params).is_err());
}

#[test]
fn test_kv_cache_pos_out_of_bounds() {
    let config = KVCacheConfig {
        num_layers: 2,
        max_seq_len: 10,
        num_heads: 4,
        head_dim: 32,
    };
    let mut cache = KVCache::new(config);
    let k = vec![0.5; 128];
    let v = vec![0.5; 128];
    let params = KVStoreParams::builder(k, v).layer(0).pos(20).build();
    assert!(cache.store(params).is_err());
}
