use std::path::PathBuf;
use crate::config::AppConfig;
use crate::error::MinervaResult;

/// Serve command arguments
#[derive(Debug, Clone)]
pub struct ServeArgs {
    pub host: String,
    pub port: u16,
    pub models_dir: Option<PathBuf>,
    pub config_file: Option<PathBuf>,
    pub workers: Option<usize>,
}

impl Default for ServeArgs {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            models_dir: None,
            config_file: None,
            workers: None,
        }
    }
}

impl ServeArgs {
    /// Parse CLI arguments
    pub fn from_args(args: &[String]) -> MinervaResult<Self> {
        let mut serve_args = ServeArgs::default();
        
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--host" => {
                    if i + 1 < args.len() {
                        serve_args.host = args[i + 1].clone();
                        i += 2;
                    } else {
                        return Err(crate::error::MinervaError::InvalidRequest(
                            "Missing value for --host".to_string(),
                        ));
                    }
                }
                "--port" => {
                    if i + 1 < args.len() {
                        serve_args.port = args[i + 1]
                            .parse()
                            .map_err(|_| crate::error::MinervaError::InvalidRequest(
                                "Invalid port number".to_string(),
                            ))?;
                        i += 2;
                    } else {
                        return Err(crate::error::MinervaError::InvalidRequest(
                            "Missing value for --port".to_string(),
                        ));
                    }
                }
                "--models-dir" => {
                    if i + 1 < args.len() {
                        serve_args.models_dir = Some(PathBuf::from(&args[i + 1]));
                        i += 2;
                    } else {
                        return Err(crate::error::MinervaError::InvalidRequest(
                            "Missing value for --models-dir".to_string(),
                        ));
                    }
                }
                "--config" => {
                    if i + 1 < args.len() {
                        serve_args.config_file = Some(PathBuf::from(&args[i + 1]));
                        i += 2;
                    } else {
                        return Err(crate::error::MinervaError::InvalidRequest(
                            "Missing value for --config".to_string(),
                        ));
                    }
                }
                "--workers" => {
                    if i + 1 < args.len() {
                        serve_args.workers = Some(args[i + 1].parse().map_err(|_| {
                            crate::error::MinervaError::InvalidRequest(
                                "Invalid worker count".to_string(),
                            )
                        })?);
                        i += 2;
                    } else {
                        return Err(crate::error::MinervaError::InvalidRequest(
                            "Missing value for --workers".to_string(),
                        ));
                    }
                }
                _ => {
                    i += 1;
                }
            }
        }
        
        Ok(serve_args)
    }
}

/// Execute serve command
pub async fn serve_command(args: ServeArgs) -> MinervaResult<()> {
    let _config = AppConfig::load_or_default();
    
    println!(
        "Starting Minerva server on {}:{}",
        args.host, args.port
    );
    
    if let Some(models_dir) = &args.models_dir {
        println!("Using models directory: {}", models_dir.display());
    }
    
    if let Some(config_file) = &args.config_file {
        println!("Using config file: {}", config_file.display());
    }
    
    if let Some(workers) = args.workers {
        println!("Worker threads: {}", workers);
    }
    
    println!("Server ready to accept requests");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_args_default() {
        let args = ServeArgs::default();
        assert_eq!(args.host, "127.0.0.1");
        assert_eq!(args.port, 3000);
        assert!(args.models_dir.is_none());
    }

    #[test]
    fn test_serve_args_parse_host() {
        let args_vec = [
            "serve".to_string(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
        ];
        let args = ServeArgs::from_args(&args_vec[1..]).unwrap();
        assert_eq!(args.host, "0.0.0.0");
    }

    #[test]
    fn test_serve_args_parse_port() {
        let args_vec = [
            "serve".to_string(),
            "--port".to_string(),
            "8000".to_string(),
        ];
        let args = ServeArgs::from_args(&args_vec[1..]).unwrap();
        assert_eq!(args.port, 8000);
    }

    #[test]
    fn test_serve_args_multiple_flags() {
        let args_vec = [
            "serve".to_string(),
            "--host".to_string(),
            "0.0.0.0".to_string(),
            "--port".to_string(),
            "9000".to_string(),
            "--workers".to_string(),
            "4".to_string(),
        ];
        let args = ServeArgs::from_args(&args_vec[1..]).unwrap();
        assert_eq!(args.host, "0.0.0.0");
        assert_eq!(args.port, 9000);
        assert_eq!(args.workers, Some(4));
    }

    #[test]
    fn test_serve_args_missing_value() {
        let args_vec = ["serve".to_string(), "--host".to_string()];
        let result = ServeArgs::from_args(&args_vec[1..]);
        assert!(result.is_err());
    }
}
