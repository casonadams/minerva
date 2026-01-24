use std::path::PathBuf;
use clap::Parser;
use crate::config::AppConfig;
use crate::error::MinervaResult;

/// Serve command arguments
#[derive(Debug, Clone, Parser)]
#[command(name = "serve", about = "Start the Minerva API server")]
pub struct ServeArgs {
    /// Server host
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,

    /// Server port
    #[arg(long, default_value = "3000")]
    pub port: u16,

    /// Models directory
    #[arg(long)]
    pub models_dir: Option<PathBuf>,

    /// Configuration file
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// Number of worker threads
    #[arg(long)]
    pub workers: Option<usize>,
}

impl Default for ServeArgs {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 3000,
            models_dir: None,
            config: None,
            workers: None,
        }
    }
}

/// Execute serve command - starts HTTP server without Tauri
pub async fn serve_command(args: ServeArgs) -> MinervaResult<()> {
    let mut config = AppConfig::load_or_default();
    
    // Override with CLI arguments if provided
    if let Some(models_dir) = &args.models_dir {
        config.models_dir = models_dir.clone();
    }
    
    println!(
        "Starting Minerva server on {}:{}",
        args.host, args.port
    );
    
    if let Some(models_dir) = &args.models_dir {
        println!("Using models directory: {}", models_dir.display());
    }
    
    if let Some(config_file) = &args.config {
        println!("Using config file: {}", config_file.display());
    }
    
    if let Some(workers) = args.workers {
        println!("Worker threads: {}", workers);
    }
    
    // Create server state with discovered models
    let server_state = crate::server::ServerState::with_discovered_models(
        config.models_dir.clone()
    )?;
    
    // Create the router
    let router = crate::server::create_server(server_state).await;
    
    // Parse socket address
    let addr = format!("{}:{}", args.host, args.port);
    let socket_addr: std::net::SocketAddr = addr.parse()
        .map_err(|e| crate::error::MinervaError::InvalidRequest(
            format!("Invalid socket address: {}", e)
        ))?;
    
    // Start the server
    let listener = tokio::net::TcpListener::bind(&socket_addr).await
        .map_err(|e| crate::error::MinervaError::InvalidRequest(
            format!("Failed to bind socket: {}", e)
        ))?;
    
    println!("Server ready to accept requests");
    
    axum::serve(listener, router).await
        .map_err(|e| crate::error::MinervaError::InvalidRequest(
            format!("Server error: {}", e)
        ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_args_default() {
        let args = ServeArgs::default();
        assert_eq!(args.host, "127.0.0.1", "Default host should be localhost");
        assert_eq!(args.port, 3000, "Default port should be 3000");
        assert!(args.models_dir.is_none(), "Models dir should be optional");
    }

    #[test]
    fn test_serve_args_from_cli() {
        // Test with clap parsing
        let args = ServeArgs::parse_from([
            "serve",
            "--host",
            "0.0.0.0",
            "--port",
            "8080",
            "--workers",
            "4",
        ]);
        assert_eq!(args.host, "0.0.0.0", "Host should parse correctly");
        assert_eq!(args.port, 8080, "Port should parse correctly");
        assert_eq!(args.workers, Some(4), "Workers should parse correctly");
    }

    #[test]
    fn test_serve_args_models_dir() {
        let args = ServeArgs::parse_from([
            "serve",
            "--models-dir",
            "/tmp/models",
        ]);
        assert_eq!(
            args.models_dir,
            Some(PathBuf::from("/tmp/models")),
            "Models dir should parse correctly"
        );
    }

    #[test]
    fn test_serve_args_config_file() {
        let args = ServeArgs::parse_from([
            "serve",
            "--config",
            "/etc/minerva.toml",
        ]);
        assert_eq!(
            args.config,
            Some(PathBuf::from("/etc/minerva.toml")),
            "Config file should parse correctly"
        );
    }

    #[test]
    fn test_serve_args_all_options() {
        let args = ServeArgs::parse_from([
            "serve",
            "--host",
            "192.168.1.1",
            "--port",
            "9000",
            "--models-dir",
            "/custom/models",
            "--config",
            "/custom/config.toml",
            "--workers",
            "8",
        ]);
        assert_eq!(args.host, "192.168.1.1", "Host should be set");
        assert_eq!(args.port, 9000, "Port should be set");
        assert_eq!(args.models_dir, Some(PathBuf::from("/custom/models")), "Models dir should be set");
        assert_eq!(args.config, Some(PathBuf::from("/custom/config.toml")), "Config should be set");
        assert_eq!(args.workers, Some(8), "Workers should be set");
    }
}
