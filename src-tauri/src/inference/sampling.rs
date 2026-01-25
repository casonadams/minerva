pub use super::greedy_sampling::sample_greedy;
pub use super::softmax_utils::{argmax, simple_random, softmax};
/// Advanced Sampling Strategies - Phase 9 Day 5 (Refactored Phase 13)
/// Production-ready token sampling: greedy, temperature, top-k, top-p (nucleus)
pub use super::temperature::{TemperatureConfig, apply_temperature};
pub use super::temperature_sampling::sample_temperature;
pub use super::top_k_sampling::{TopKConfig, sample_top_k};
pub use super::top_p_sampling::{TopPConfig, sample_top_p};
