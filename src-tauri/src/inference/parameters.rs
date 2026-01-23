use super::GenerationConfig;
use crate::error::{MinervaError, MinervaResult};
use crate::models::ChatCompletionRequest;

/// Parse and validate generation parameters from a chat completion request
#[derive(Debug)]
#[allow(dead_code)]
pub struct ParameterParser;

impl ParameterParser {
    /// Extract and validate generation config from request
    #[allow(dead_code)]
    pub fn from_request(req: &ChatCompletionRequest) -> MinervaResult<GenerationConfig> {
        let mut config = GenerationConfig::default();

        // Parse temperature
        if let Some(temp) = req.temperature {
            if !(0.0..=2.0).contains(&temp) {
                return Err(MinervaError::InvalidRequest(format!(
                    "temperature must be between 0.0 and 2.0, got {}",
                    temp
                )));
            }
            config.temperature = temp;
        }

        // Parse top_p
        if let Some(top_p) = req.top_p {
            if !(0.0..=1.0).contains(&top_p) {
                return Err(MinervaError::InvalidRequest(format!(
                    "top_p must be between 0.0 and 1.0, got {}",
                    top_p
                )));
            }
            config.top_p = top_p;
        }

        // Parse frequency_penalty
        if let Some(freq_penalty) = req.frequency_penalty {
            if !(-2.0..=2.0).contains(&freq_penalty) {
                return Err(MinervaError::InvalidRequest(format!(
                    "frequency_penalty must be between -2.0 and 2.0, got {}",
                    freq_penalty
                )));
            }
            // Map frequency_penalty to repeat_penalty
            config.repeat_penalty = 1.0 + (freq_penalty / 10.0);
        }

        // Parse max_tokens
        if let Some(max_tokens) = req.max_tokens {
            if !(1..=32768).contains(&max_tokens) {
                return Err(MinervaError::InvalidRequest(format!(
                    "max_tokens must be between 1 and 32768, got {}",
                    max_tokens
                )));
            }
            config.max_tokens = max_tokens;
        }

        // Validate final config
        config.validate()?;

        Ok(config)
    }

    /// Build request summary for logging
    #[allow(dead_code)]
    pub fn summarize_request(req: &ChatCompletionRequest) -> String {
        let temp = req.temperature.unwrap_or(0.7);
        let max_tok = req.max_tokens.unwrap_or(512);
        let stream = req.stream.unwrap_or(false);

        format!(
            "model={}, messages={}, temp={:.1}, max_tokens={}, stream={}",
            req.model,
            req.messages.len(),
            temp,
            max_tok,
            stream
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ChatMessage;

    #[derive(Default)]
    struct TestRequestParams {
        temperature: Option<f32>,
        top_p: Option<f32>,
        max_tokens: Option<usize>,
        frequency_penalty: Option<f32>,
    }

    fn make_request(params: TestRequestParams) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            temperature: params.temperature,
            max_tokens: params.max_tokens,
            stream: None,
            top_p: params.top_p,
            frequency_penalty: params.frequency_penalty,
            presence_penalty: None,
        }
    }

    #[test]
    fn test_parameter_parser_defaults() {
        let req = make_request(TestRequestParams::default());
        let config = ParameterParser::from_request(&req).unwrap();

        assert_eq!(config.temperature, 0.7);
        assert_eq!(config.top_p, 0.9);
        assert_eq!(config.max_tokens, 512);
    }

    #[test]
    fn test_parameter_parser_custom_values() {
        let params = TestRequestParams {
            temperature: Some(0.5),
            top_p: Some(0.8),
            max_tokens: Some(1024),
            frequency_penalty: None,
        };
        let req = make_request(params);
        let config = ParameterParser::from_request(&req).unwrap();

        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.top_p, 0.8);
        assert_eq!(config.max_tokens, 1024);
    }

    #[test]
    fn test_parameter_parser_invalid_temperature() {
        let params = TestRequestParams {
            temperature: Some(3.0),
            ..Default::default()
        };
        let req = make_request(params);
        let result = ParameterParser::from_request(&req);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("temperature must be between"));
    }

    #[test]
    fn test_parameter_parser_invalid_top_p() {
        let params = TestRequestParams {
            top_p: Some(1.5),
            ..Default::default()
        };
        let req = make_request(params);
        let result = ParameterParser::from_request(&req);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("top_p"));
    }

    #[test]
    fn test_parameter_parser_invalid_max_tokens() {
        let params = TestRequestParams {
            max_tokens: Some(0),
            ..Default::default()
        };
        let req = make_request(params);
        let result = ParameterParser::from_request(&req);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("max_tokens must be"));
    }

    #[test]
    fn test_parameter_parser_frequency_penalty() {
        let params = TestRequestParams {
            frequency_penalty: Some(0.0),
            ..Default::default()
        };
        let req = make_request(params);
        let config = ParameterParser::from_request(&req).unwrap();

        // frequency_penalty 0.0 maps to repeat_penalty 1.0
        assert_eq!(config.repeat_penalty, 1.0);
    }

    #[test]
    fn test_parameter_parser_frequency_penalty_positive() {
        let params = TestRequestParams {
            frequency_penalty: Some(1.0),
            ..Default::default()
        };
        let req = make_request(params);
        let config = ParameterParser::from_request(&req).unwrap();

        // frequency_penalty 1.0 maps to repeat_penalty ~1.1
        assert!((config.repeat_penalty - 1.1).abs() < 0.01);
    }

    #[test]
    fn test_parameter_parser_invalid_frequency_penalty() {
        let params = TestRequestParams {
            frequency_penalty: Some(3.0),
            ..Default::default()
        };
        let req = make_request(params);
        let result = ParameterParser::from_request(&req);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("frequency_penalty"));
    }

    #[test]
    fn test_parameter_parser_summarize() {
        let params = TestRequestParams {
            temperature: Some(0.8),
            max_tokens: Some(2048),
            ..Default::default()
        };
        let req = make_request(params);
        let summary = ParameterParser::summarize_request(&req);

        assert!(summary.contains("model=test"));
        assert!(summary.contains("temp=0.8"));
        assert!(summary.contains("max_tokens=2048"));
        assert!(summary.contains("stream=false"));
    }

    #[test]
    fn test_parameter_parser_summarize_streaming() {
        let mut req = make_request(TestRequestParams::default());
        req.stream = Some(true);
        let summary = ParameterParser::summarize_request(&req);

        assert!(summary.contains("stream=true"));
    }
}
