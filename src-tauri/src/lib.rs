pub mod commands;
pub mod config;
pub mod error;
pub mod error_recovery;
pub mod inference;
pub mod models;
pub mod server;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app_config = config::AppConfig::load_or_default();

    // Ensure models directory exists on startup
    if let Err(e) = app_config.ensure_models_dir() {
        eprintln!("Warning: Failed to create models directory: {}", e);
    }

    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .manage(commands::AppState {
            config: std::sync::Mutex::new(app_config),
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_config,
            commands::set_models_directory,
            commands::get_models_directory,
            commands::list_discovered_models,
            commands::load_model_file,
            commands::ensure_models_directory,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
