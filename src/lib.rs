pub mod args;
pub mod inference;
pub mod model;
pub mod postprocessing;
pub mod preprocessing;

use anyhow::{Context, Result};
use image::{GenericImageView, RgbImage};

use args::Args;
use inference::DepthEstimator;
use model::{ModelConfig, ModelType};

pub fn run_depth_anything(args: Args) -> Result<(), anyhow::Error> {
    // Load image
    let dyn_img = image::open(&args.input)
        .with_context(|| format!("Failed to open image: {}", &args.input))?;
    let (orig_w, orig_h) = dyn_img.dimensions();
    let rgb8: RgbImage = dyn_img.to_rgb8();

    // Prepare model config
    let model_config = ModelConfig::from_type(args.model_type)?;
    println!(
        "Using {} model: {}",
        match args.model_type {
            ModelType::Static => "static",
            ModelType::Dynamic => "dynamic",
        },
        model_config.path.display()
    );

    // Create estimator and run inference
    let mut estimator = DepthEstimator::new(
        model_config,
        args.threads,
        args.use_cuda,
        args.use_tensorrt,
        args.use_directml,
    )?;
    let depth_map = estimator.estimate_depth(&rgb8)?;

    // Post-process output
    let final_img = if args.resize_to_input {
        let (img_w, img_h) = (depth_map.width(), depth_map.height());
        if img_w != orig_w || img_h != orig_h {
            image::imageops::resize(
                &depth_map,
                orig_w,
                orig_h,
                image::imageops::FilterType::CatmullRom,
            )
        } else {
            depth_map
        }
    } else {
        depth_map
    };

    // Write output to disk
    final_img
        .save(&args.output)
        .with_context(|| format!("Failed to save {}", &args.output))?;
    println!(
        "Saved depth map: {} ({}x{}, 16-bit)",
        &args.output,
        final_img.width(),
        final_img.height()
    );
    Ok(())
}
