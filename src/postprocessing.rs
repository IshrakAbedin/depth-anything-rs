use anyhow::Result;
use image::{ImageBuffer, Luma};
use ndarray::Array2;

/// Convert HxW f32 array to 16-bit grayscale image with min-max normalization.
pub fn to_luma16(depth: Array2<f32>, w: u32, h: u32) -> Result<ImageBuffer<Luma<u16>, Vec<u16>>> {
    let min = depth.iter().copied().fold(f32::INFINITY, f32::min);
    let max = depth.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom = (max - min).max(1e-6);

    let mut buf = Vec::<u16>::with_capacity((w * h) as usize);
    for v in depth.iter() {
        let norm = ((*v - min) / denom).clamp(0.0, 1.0);
        let q = (norm * 65535.0).round() as u16;
        buf.push(q);
    }

    let img = ImageBuffer::<Luma<u16>, _>::from_vec(w, h, buf)
        .ok_or_else(|| anyhow::anyhow!("Failed to build Luma16 image"))?;
    Ok(img)
}
