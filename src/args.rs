use clap::{Parser, ValueEnum};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Model type to use
    #[arg(short, long, value_enum, default_value_t = ModelType::Static)]
    pub model_type: ModelType,

    /// Input image path
    #[arg(short, long)]
    pub input: String,

    /// Output depth map path (16-bit PNG)
    #[arg(short, long, default_value="depth.png")]
    pub output: String,

    /// Resize output back to the input image dimensions before saving
    #[arg(long, default_value_t = true)]
    pub resize_to_input: bool,

    /// Number of intra-op threads for ORT
    #[arg(long, default_value_t = 4)]
    pub threads: usize,

    /// Try to register CUDA EP (requires building with --features cuda)
    #[arg(long, default_value_t = false)]
    pub use_cuda: bool,

    /// Try to register TensorRT EP (requires --features tensorrt). If both TRT and CUDA enabled,
    /// TensorRT is preferred first, then CUDA as fallback.
    #[arg(long, default_value_t = false)]
    pub use_tensorrt: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
pub enum ModelType {
    /// Use static 518x518 model (faster, optimized for GPU)
    Static,
    /// Use dynamic model (flexible input size, higher quality)
    Dynamic,
}
