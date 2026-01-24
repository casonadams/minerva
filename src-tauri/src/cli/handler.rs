//! CLI command handlers

use super::types::ServeArgs;
use crate::config::AppConfig;
use crate::error::MinervaResult;

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
