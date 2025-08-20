use anyhow::Result;
use image::{Pixel, RgbImage, imageops::FilterType};
use ndarray::Array4;

/// Depth Anything v2 / DPT preprocessing:
/// - If `static_resize`: resize to exactly (size,size) (squash).
/// - Else: keep aspect ratio, fit within (size,size), then round H and W down to multiples of 14.
pub fn preprocess(rgb8: &RgbImage, size: u32, static_resize: bool) -> Result<Array4<f32>> {
    let (w, h) = (rgb8.width(), rgb8.height());

    let (new_w, new_h) = if static_resize {
        (size, size)
    } else {
        let scale = (size as f32 / w as f32).min(size as f32 / h as f32);
        let mut nw = ((w as f32) * scale).round() as u32;
        let mut nh = ((h as f32) * scale).round() as u32;
        // ensure multiple-of-14 (Depth Anything v2 dynamic ONNX requires H,W % 14 == 0)
        nw = (nw / 14).max(1) * 14;
        nh = (nh / 14).max(1) * 14;
        (nw, nh)
    };

    let resized = image::imageops::resize(rgb8, new_w, new_h, FilterType::CatmullRom);

    // Convert to CHW float32 and normalize: (x/255 - mean) / std
    // ImageNet stats (from HF DPT processor config for Depth Anything v2)
    let mean = [0.485f32, 0.456f32, 0.406f32];
    let std = [0.229f32, 0.224f32, 0.225f32];

    let (nw, nh) = (resized.width() as usize, resized.height() as usize);
    let mut arr = Array4::<f32>::zeros((1, 3, nh, nw));

    for y in 0..nh {
        for x in 0..nw {
            let p = resized.get_pixel(x as u32, y as u32).channels();
            // channels are [R,G,B] u8
            for c in 0..3 {
                let v = p[c] as f32 / 255.0;
                let n = (v - mean[c]) / std[c];
                arr[(0, c, y, x)] = n;
            }
        }
    }

    Ok(arr)
}
