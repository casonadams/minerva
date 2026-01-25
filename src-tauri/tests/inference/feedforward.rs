use minerva_lib::inference::activation::Activation;
use minerva_lib::inference::feedforward::{feedforward, FeedforwardConfig, FeedforwardWeights};

#[test]
fn test_feedforward_shapes() {
    let seq_len = 2;
    let hidden_size = 4;
    let intermediate_size = 8;

    let input = vec![0.1; seq_len * hidden_size];
    let w_up = vec![0.2; hidden_size * intermediate_size];
    let w_down = vec![0.1; intermediate_size * hidden_size];

    let config = FeedforwardConfig {
        seq_len,
        hidden_size,
        intermediate_size,
        activation: Activation::GELU,
    };

    let weights = FeedforwardWeights {
        up: &w_up,
        down: &w_down,
    };
    let output = feedforward(&input, &weights, &config).unwrap();
    assert_eq!(output.len(), seq_len * hidden_size);
}

#[test]
fn test_feedforward_all_activations() {
    let seq_len = 1;
    let hidden_size = 2;
    let intermediate_size = 4;

    let input = vec![1.0; seq_len * hidden_size];
    let w_up = vec![0.5; hidden_size * intermediate_size];
    let w_down = vec![0.5; intermediate_size * hidden_size];

    let weights = FeedforwardWeights {
        up: &w_up,
        down: &w_down,
    };

    for activation in [Activation::GELU, Activation::SiLU, Activation::ReLU] {
        let config = FeedforwardConfig {
            seq_len,
            hidden_size,
            intermediate_size,
            activation,
        };
        let result = feedforward(&input, &weights, &config);
        assert!(result.is_ok());
    }
}

#[test]
fn test_feedforward_input_mismatch() {
    let config = FeedforwardConfig {
        seq_len: 2,
        hidden_size: 4,
        intermediate_size: 8,
        activation: Activation::GELU,
    };

    let input = vec![0.1; 5]; // Wrong size
    let w_up = vec![0.2; 4 * 8];
    let w_down = vec![0.1; 8 * 4];

    let weights = FeedforwardWeights {
        up: &w_up,
        down: &w_down,
    };
    let result = feedforward(&input, &weights, &config);
    assert!(result.is_err());
}
