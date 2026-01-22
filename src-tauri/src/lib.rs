mod server;
mod models;
mod error;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            commands::start_server,
            commands::stop_server,
            commands::get_models,
            commands::server_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

mod commands {
    #[tauri::command]
    pub fn start_server() -> Result<String, String> {
        Ok("Server started".to_string())
    }

    #[tauri::command]
    pub fn stop_server() -> Result<String, String> {
        Ok("Server stopped".to_string())
    }

    #[tauri::command]
    pub fn get_models() -> Result<Vec<String>, String> {
        Ok(vec!["model1".to_string()])
    }

    #[tauri::command]
    pub fn server_status() -> Result<String, String> {
        Ok("running".to_string())
    }
}
