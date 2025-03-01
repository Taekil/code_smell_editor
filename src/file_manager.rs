// fileNamager.rs

use rfd::FileDialog;

pub struct FileManager;

impl FileManager {
    pub fn new() -> Self {
        FileManager
    }

    pub fn upload_file(&self) -> Result<String, String> {
        if let Some(file) = FileDialog::new().pick_file() {
            let file_path = file.display().to_string();
            match std::fs::read_to_string(&file_path) {
                Ok(content) => Ok(content),
                Err(_) => Err("Failed reading file..".to_string()),
            }
        } else {
            Err("No file selected..".to_string())
        }
     }

     pub fn save_file(&self, content: &str) -> String {
        if let Some(file) = FileDialog::new().save_file() {
            let file_path = file.display().to_string();
            match std::fs::write(&file_path, content) {
                Ok(_) => "File Saved Successfully".to_string(),
                Err(err) => format!("Save Failed: {}", err),
            }
        } else {
            "No file selected for saving.".to_string()
        }
     }
}