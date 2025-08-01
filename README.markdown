# Text-to-3D Generation Pipeline

This project implements a two-stage pipeline for generating high-quality 3D models from text descriptions using Stable Diffusion XL (SDXL) or Stable Diffusion 1.5 (fallback) for image generation and Trellis for 3D model creation, deployed on Modal. The process is automated using a batch file that coordinates both stages.

## Prerequisites

- Modal account and CLI installed (`pip install modal-client`)
- Python with conda environment (default name: `modal_env`)
- Blender 4.4+ installed
- CUDA 11.8-capable GPU (architecture 86) recommended on Modal (A10G used)
- Required Python packages:
  ```bash
  modal-client
  torch==2.3.1+cu118 torchvision==0.18.1+cu118
  diffusers==0.21.4 transformers==4.33.0 accelerate==0.22.0 safetensors==0.3.3 huggingface-hub==0.16.4
  pillow==10.0.0 imageio imageio-ffmpeg opencv-python scikit-image scipy
  trimesh kaolin pytorch-lightning==2.0.4
  flash-attn==2.6.3 xformers==0.0.27+cu118 spconv-cu118
  pyyaml pandas einops tqdm kornia omegaconf jaxtyping wandb Cython
  ```
- SDXL or Stable Diffusion 1.5 model weights (`runwayml/stable-diffusion-v1-5` for fallback) accessible via Modal
- Trellis model weights (`microsoft/TRELLIS-image-large`) configured on Modal

## Pipeline Overview

### Stage 1: Image Generation
The first stage uses SDXL (or Stable Diffusion 1.5 as fallback) to generate a high-quality 2D image from a text prompt on Modal:

- Processes the text prompt through SDXL or SD 1.5
- Generates a detailed image (1024x1024 for SDXL, 512x512 for SD 1.5)
- Saves the image as an intermediate PNG file

### Stage 2: 3D Model Generation and Optimization
The second stage uses Trellis to generate a 3D model from the image and Blender to optimize it:

- Trellis converts the image to a 3D mesh, exporting an intermediate GLB file
- Blender simplifies the mesh, creates smart UV maps, and applies smooth shading
- Exports the final optimized GLB file

## Usage

1. Install and set up Modal to use the mesh generation code.
2. Run the batch file:
   ```bash
   auto.bat
   ```

## File Structure

```
├── auto.bat              # Main automation script
├── TReS.py               # Modal app for SDXL/SD 1.5 and Trellis
├── finisher.py           # Stage 2: Mesh optimization
├── interim_output/       # Stores images and dense meshes
│   ├── spaceship.png
│   └── spaceship.glb
├── Final/                # Stores final optimized models
│   └── spaceship_final.glb
```

## Output Files

- Intermediate image: `interim_output/spaceship.png`
- Intermediate dense mesh: `interim_output/spaceship.glb`
- Final optimized model: `Final/spaceship_final.glb`

## Error Handling

The batch file includes error checking at each stage:

- Validates Modal app deployment and connectivity
- Falls back to Stable Diffusion 1.5 if SDXL fails
- Checks for successful image and mesh generation
- Verifies final file creation in Stage 2

## Technical Details

### Stage 1 Parameters

**SDXL (primary):**
- Guidance scale: 7.5
- Number of inference steps: 50
- Output resolution: 1024x1024
- FP16 precision enabled

**Stable Diffusion 1.5 (fallback):**
- Guidance scale: 7.5
- Number of inference steps: 25
- Output resolution: 512x512
- FP16 precision with sequential CPU offload

### Stage 2 Parameters

**Trellis:**
- Model: `microsoft/TRELLIS-image-large`
- Sparse structure sampler: 20 steps, CFG strength 8.5
- SLAT sampler: 15 steps, CFG strength 4.0
- Formats: mesh, Gaussian
- GLB export: simplify=0.97, texture_size=2048

**

Blender:**
- Smart UV unwrapping (66° angle limit)
- Smooth shading
- Target face count customizable (default: 40,000)
- Island margin: 0.02

## Troubleshooting

### If Stage 1 fails:
- Verify Modal CLI setup and authentication (`modal token set`)
- Check model weights availability (`runwayml/stable-diffusion-v1-5` or SDXL via `huggingface-hub`)
- Ensure sufficient Modal credits or GPU availability (A10G, CUDA 11.8, arch 86)

### If Stage 2 fails:
- Verify Trellis installation in `app.py` (check `--basic` setup logs)
- Check Blender installation path
- Ensure intermediate files exist (`output/spaceship.png`, `output/spaceship.glb`)
- Ensure sufficient disk space