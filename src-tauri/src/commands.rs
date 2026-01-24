pub mod model_commands;

use crate::config::AppConfig;
use crate::models::ModelInfo;
use std::path::PathBuf;
use std::sync::Mutex;

/// Application state for Tauri commands
pub struct AppState {
    pub config: Mutex<AppConfig>,
}

/// Get application configuration
#[tauri::command]
pub fn get_config(state: tauri::State<'_, AppState>) -> Result<AppConfig, String> {
    state
        .config
        .lock()
        .map_err(|e| format!("Failed to lock config: {}", e))
        .map(|cfg| cfg.clone())
}

/// Set models directory path
#[tauri::command]
pub fn set_models_directory(state: tauri::State<'_, AppState>, path: String) -> Result<(), String> {
    let path_buf = PathBuf::from(&path);

    if !path_buf.exists() {
        return Err(format!("Directory does not exist: {}", path));
    }

    let mut config = state
        .config
        .lock()
        .map_err(|e| format!("Failed to lock config: {}", e))?;

    config.models_dir = path_buf;
    config
        .save()
        .map_err(|e| format!("Failed to save config: {}", e))
}

/// Get models directory path
#[tauri::command]
pub fn get_models_directory(state: tauri::State<'_, AppState>) -> Result<String, String> {
    state
        .config
        .lock()
        .map_err(|e| format!("Failed to lock config: {}", e))
        .map(|cfg| cfg.models_dir.to_string_lossy().to_string())
}

/// Discover and list all models in the models directory
#[tauri::command]
pub fn list_discovered_models(state: tauri::State<'_, AppState>) -> Result<Vec<ModelInfo>, String> {
    let config = state
        .config
        .lock()
        .map_err(|e| format!("Failed to lock config: {}", e))?;

    let loader = crate::models::loader::ModelLoader::new(config.models_dir.clone());

    loader
        .discover_models()
        .map_err(|e| format!("Failed to discover models: {}", e))
}

/// Load a specific model by path
#[tauri::command]
pub fn load_model_file(model_path: String) -> Result<ModelInfo, String> {
    let path = PathBuf::from(&model_path);

    if !path.exists() {
        return Err(format!("Model file not found: {}", model_path));
    }

    if path.extension().and_then(|s| s.to_str()) != Some("gguf") {
        return Err("Only GGUF files are supported".to_string());
    }

    let loader = crate::models::loader::ModelLoader::new(
        path.parent()
            .unwrap_or_else(|| std::path::Path::new("."))
            .to_path_buf(),
    );

    loader
        .load_model(&path)
        .map_err(|e| format!("Failed to load model: {}", e))
}

/// Ensure models directory exists and create if necessary
#[tauri::command]
pub fn ensure_models_directory(state: tauri::State<'_, AppState>) -> Result<String, String> {
    let config = state
        .config
        .lock()
        .map_err(|e| format!("Failed to lock config: {}", e))?;

    config
        .ensure_models_dir()
        .map_err(|e| format!("Failed to create models directory: {}", e))?;

    Ok(config.models_dir.to_string_lossy().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_state_creation() {
        let config = AppConfig::default();
        let state = AppState {
            config: Mutex::new(config),
        };

        assert!(state.config.lock().is_ok());
    }
}
