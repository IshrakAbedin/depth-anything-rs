# Depth-Anything-Rust

A command-line tool to estimate depth maps using [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

## Building and running the project

### Building

You can simply build the project using `cargo build` for debug mode and `cargo build -r` for release mode.

> It supports optional features for CUDA and TensorRT, but not tested (if you compile with any one of those, consult the *Running* section to learn how to turn on CUDA or TensorRT for inference from the CLI). The support for DirectML is lightly tested and found to be working. The DirectML feature also needs to be turned on for inference from the CLI.

However, you need to supply the application with the `ONNX` models of the static and dynamic versions of Depth-Anything-V2 by putting them under the root directory of your project inside the `./models/` folder, or in the residing directory of your executable under the `./models/` folder as `depth_anything_v2_vitb.onnx` (static) and `depth_anything_v2_vitb_dynamic.onnx` (dynamic). If you are using a larger or smaller model, you can edit the paths in the [`model.rs`](./src/model.rs) file.

I fetched the `ONNX` files from [the ONNX Release in GitHub](https://github.com/fabio-sim/Depth-Anything-ONNX/releases/tag/v2.0.0)

### Running
Once you have the project compiled and the models placed properly, you can run it either using

```sh
# Release mode running, debug will not have the -r
cargo run -r -- [OPTIONS]
```

or, if you have taken the binary out then

```sh
# Windows will have .exe after the app name
depth-anything-rs [OPTIONS]
```

The options are:

```
Options:
  -m, --model-type <MODEL_TYPE>
          Model type to use

          Possible values:
          - static:  Use static 518x518 model (faster, optimized for GPU)
          - dynamic: Use dynamic model (flexible input size, higher quality)

          [default: static]

  -i, --input <INPUT>
          Input image path

  -o, --output <OUTPUT>
          Output depth map path (16-bit PNG)

          [default: depth.png]

      --resize-to-input
          Resize output back to the input image dimensions before saving

      --threads <THREADS>
          Number of intra-op threads for ORT

          [default: 4]

      --use-cuda
          Try to register CUDA EP (requires building with --features cuda)

      --use-tensorrt
          Try to register TensorRT EP (requires --features tensorrt). If both TRT and CUDA enabled, TensorRT is preferred first, then CUDA as fallback

      --use-directml
          Try to register DirectML EP (requires building with --features directml)

  -d, --device-id <DEVICE_ID>
          Device (GPU) ID to be used with CUDA, TensorRT, or DirectML

          [default: 0]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

Only the `--input`/`-i` argument is mandatory. 

## Note
> I have let the `ort` crate handle the ONNX Runtime for both CPU and with CUDA/TensorRT/DirectML acceleration. If it creates problem, you might want to look into how to link your own dynamic libraries for ONNX.
