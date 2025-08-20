pub use crate::args::ModelType;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub struct ModelConfig {
    pub path: PathBuf,
    pub target_size: u32,
    pub static_resize: bool,
}

impl ModelConfig {
    pub fn from_type(model_type: ModelType) -> Result<Self> {
        let (filename, target_size, static_resize) = match model_type {
            ModelType::Static => ("depth_anything_v2_vitb.onnx", 518, true),
            ModelType::Dynamic => ("depth_anything_v2_vitb_dynamic.onnx", 518, false),
        };

        let path = find_model_path(filename)?;

        Ok(ModelConfig {
            path,
            target_size,
            static_resize,
        })
    }
}

fn find_model_path(filename: &str) -> Result<PathBuf> {
    // First try: models folder relative to executable
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(exe_dir) = exe_path.parent() {
            let model_path = exe_dir.join("models").join(filename);
            if model_path.exists() {
                return Ok(model_path);
            }
        }
    }

    // Second try: models folder in current working directory
    let cwd_model_path = Path::new("models").join(filename);
    if cwd_model_path.exists() {
        return Ok(cwd_model_path);
    }

    // If neither location works, return the expected path with an error message
    Err(anyhow::anyhow!(
        "Model file '{}' not found. Searched in:\n\
         1. [executable_dir]/models/{}\n\
         2. [current_dir]/models/{}\n\
         Please ensure the model file exists in one of these locations.",
        filename,
        filename,
        filename
    ))
}
