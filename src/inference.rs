use anyhow::{Context, Result, bail};
use image::{ImageBuffer, Luma, RgbImage};
use ndarray::Array2;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};

use crate::model::ModelConfig;
use crate::postprocessing::to_luma16;
use crate::preprocessing::preprocess;

pub struct DepthEstimator {
    session: Session,
    config: ModelConfig,
}

impl DepthEstimator {
    pub fn new(
        config: ModelConfig,
        threads: usize,
        use_cuda: bool,
        use_tensorrt: bool,
        use_directml: bool,
        device_id: i32,
    ) -> Result<Self> {
        let mut builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(threads)?;

        // Optional execution providers (feature-gated at compile time)
        #[cfg(feature = "tensorrt")]
        if use_tensorrt {
            use ort::execution_providers::TensorRTExecutionProvider;
            let trt = TensorRTExecutionProvider::default()
                .with_device_id(device_id)
                .build()
                .error_on_failure();
            builder = builder.with_execution_providers([trt])?;
        }

        #[cfg(feature = "cuda")]
        if use_cuda {
            use ort::execution_providers::CUDAExecutionProvider;
            let cuda = CUDAExecutionProvider::default()
                .with_device_id(device_id)
                .build()
                .error_on_failure();
            builder = builder.with_execution_providers([cuda])?;
        }

        #[cfg(feature = "directml")]
        if use_directml {
            use ort::execution_providers::DirectMLExecutionProvider;
            let dml = DirectMLExecutionProvider::default()
                .with_device_id(device_id)
                .build()
                .error_on_failure();
            builder = builder.with_execution_providers([dml])?;
        }

        #[cfg(not(any(feature = "cuda", feature = "tensorrt", feature = "directml")))]
        {
            if use_cuda || use_tensorrt || use_directml {
                eprintln!(
                    "Note: you passed --use-cuda/--use-tensorrt/--use-directml but the binary was not built with those features."
                );
            }
        }

        let session = builder
            .commit_from_file(&config.path)
            .with_context(|| format!("Failed to load ONNX model: {}", config.path.display()))?;

        // Debug: Print input and output names
        let input_names: Vec<_> = session.inputs.iter().map(|i| &i.name).collect();
        let output_names: Vec<_> = session.outputs.iter().map(|o| &o.name).collect();
        println!("Model input names: {:?}", input_names);
        println!("Model output names: {:?}", output_names);

        Ok(DepthEstimator { session, config })
    }

    pub fn estimate_depth(&mut self, rgb8: &RgbImage) -> Result<ImageBuffer<Luma<u16>, Vec<u16>>> {
        // ---- Preprocess
        let proc = preprocess(rgb8, self.config.target_size, self.config.static_resize)?;

        // ---- Inference
        let input_name = self.session.inputs[0].name.clone();
        let input_tensor = Tensor::from_array(proc)?;
        let outputs = self
            .session
            .run(ort::inputs![input_name.as_str() => input_tensor])
            .context("ORT inference failed")?;

        let depth_output = outputs[0].try_extract_array::<f32>()?;
        let shape = depth_output.shape();

        println!("Output shape: {:?}", shape);

        // Convert view to owned array first
        let depth_owned = depth_output.to_owned();

        // Handle both 3D [1, H, W] and 4D [1, 1, H, W] outputs
        let depth_hw: Array2<f32> = match shape.len() {
            3 => {
                // Shape is [1, H, W] - remove batch dimension
                if shape[0] != 1 {
                    bail!("Unexpected batch size {} (expected 1)", shape[0]);
                }
                let depth_3d = depth_owned.into_dimensionality::<ndarray::Ix3>()?;
                depth_3d.index_axis(ndarray::Axis(0), 0).to_owned()
            }
            4 => {
                // Shape is [1, 1, H, W] - remove both batch and channel dimensions
                if shape[0] != 1 || shape[1] != 1 {
                    bail!("Unexpected shape {:?} (expected [1,1,H,W])", shape);
                }
                let depth_4d = depth_owned.into_dimensionality::<ndarray::Ix4>()?;
                depth_4d
                    .index_axis(ndarray::Axis(0), 0) // remove batch dim
                    .index_axis(ndarray::Axis(0), 0) // remove channel dim
                    .to_owned()
            }
            _ => {
                bail!(
                    "Unexpected output dimensionality {} (expected 3D [1,H,W] or 4D [1,1,H,W])",
                    shape.len()
                );
            }
        };

        let (out_h, out_w) = (depth_hw.shape()[0] as u32, depth_hw.shape()[1] as u32);

        // ---- Postprocess: min-max normalize to 16-bit
        to_luma16(depth_hw, out_w, out_h)
    }
}
