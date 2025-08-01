from pathlib import Path
import subprocess
import modal
import os

# Define the Modal stub and image
stub = modal.App("TreS-3D-Generation")
image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu20.04",add_python="3.10")
    .run_commands(
        "export DEBIAN_FRONTEND=noninteractive",
        "export TZ=Asia/Kolkata",
        "ln -fs /usr/share/zoneinfo/$TZ /etc/localtime",
        "echo $TZ > /etc/timezone",
        "apt-get update",
        "apt-get install -y tzdata"
    )
    .apt_install(
        "software-properties-common", "git", "build-essential",
        "ninja-build", "cmake", "libxext6", "libxrender-dev", "libgl1", "curl","bash","ffmpeg","clang","libglm-dev"
    )
    .run_commands(
        "echo 'export CUDA_HOME=/usr/local/cuda' >> /etc/environment",
        "echo 'export PATH=$CUDA_HOME/bin:$PATH' >> /etc/environment",
        "echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> /etc/environment"
    )
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:$PATH",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:$LD_LIBRARY_PATH",
        "TCNN_CUDA_ARCHITECTURES": "86"  # Add TRELLIS to Python path
    })
    .pip_install(
        # Use PyTorch 2.3.1 for better xformers compatibility
        "torch==2.3.1+cu118",
        "torchvision==0.18.1+cu118", 
        "diffusers==0.21.4",
        "transformers==4.33.0",
        "accelerate==0.22.0",
        "safetensors==0.3.3",
        "huggingface-hub==0.16.4",
        "pillow==10.0.0",
        "imageio",
        "imageio-ffmpeg",
        "trimesh",
        "pyyaml",
        "pandas",
        "pytorch-lightning==2.0.4",
        "scikit-image",
        "scipy",
        "opencv-python",
        "einops",
        "tqdm",
        "kornia",
        "omegaconf",
        "jaxtyping",
        "wandb",
        "Cython",
        extra_index_url="https://download.pytorch.org/whl/cu118"
    )
    .run_commands(
        # Install build dependencies
        "apt-get install -y libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev libgles2-mesa-dev libgl1-mesa-glx",
        
        # Set environment variables for compilation
        "export CUDA_HOME=/usr/local/cuda",
        "export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/gcc'",
        
        # Install nvdiffrast with specific settings
        "git clone https://github.com/NVlabs/nvdiffrast.git",
        "cd nvdiffrast && python setup.py install",
        "cd ..",
        
        # Try to install diffoctreerast, skip if it fails
        "git clone https://github.com/NVlabs/nvdiffrast || echo 'diffoctreerast clone failed, skipping'",
        "cd diffoctreerast && pip install -e . || echo 'diffoctreerast install failed, skipping'",
        "cd .. || true",
        
        # Install other dependencies manually
        "pip install flash-attn==2.6.3 --no-build-isolation",
        "pip install xformers==0.0.27+cu118 --index-url https://download.pytorch.org/whl/cu118",

        "git clone https://github.com/autonomousvision/mip-splatting.git /mip-splatting",
        "cd mip-splatting && export TORCH_CUDA_ARCH_LIST='7.5;8.0;8.6' && pip install /mip-splatting/submodules/diff-gaussian-rasterization && pip install /mip-splatting/submodules/simple-knn/",
        
        # Install spconv
        "pip install spconv-cu118",
        
        # Install kaolin with fallback
        "git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git",
        "cd kaolin && python setup.py install",
        # Clone TRELLIS
        "git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git",
        "cd TRELLIS && chmod +x setup.sh",
        # Install only basic and mipgaussian (skip problematic dependencies)
        "cd TRELLIS && ./setup.sh --basic || ./setup.sh --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast",
        
        # Install TRELLIS as a package
        "cd TRELLIS",
        
        # Verify critical installations
        "python -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'",
        "python -c 'import nvdiffrast; print(f\"nvdiffrast imported successfully\")' || echo 'nvdiffrast not available'",
        "python -c 'import xformers; print(f\"xformers: {xformers.__version__}\")'",
        "python -c 'import flash_attn; print(\"flash_attn imported successfully\")' || echo 'flash_attn not available'",
        "python -c 'import spconv; print(\"spconv imported successfully\")' || echo 'spconv not available'",
    
    )
    .env({"PYTHONPATH": "/TRELLIS"})
)

@stub.function(
    image=image,
    gpu="A10G",
    timeout=1800
)
def generate_3dModel(local_data: bytes) -> bytes:
    import sys
    import os
    
    # Ensure TRELLIS is in Python path
    sys.path.insert(0, '/TRELLIS')
    os.chdir('/TRELLIS')
    
    from PIL import Image
    import io
    from trellis.pipelines import TrellisImageTo3DPipeline
    from trellis.utils import postprocessing_utils
    from trellis.representations.radiance_field import Strivec
    
    os.environ['ATTN_BACKEND'] = 'xformers' 

    try:
        image_file = io.BytesIO(local_data)
        
        # Initialize pipeline
        pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
        pipeline.to("cuda")
        
        load_image = Image.open(image_file)
        
        # Generate 3D model
        output = pipeline.run(
            load_image,
            seed=42,
            formats=["mesh","gaussian"],
            sparse_structure_sampler_params={
                "steps": 20,
                "cfg_strength": 8.5
            },
            slat_sampler_params={"steps": 15,"cfg_strength": 4.0},            
            preprocess_image=True
        )

        # Convert to GLB
        glb_maker = postprocessing_utils.to_glb(output['gaussian'][0],output['mesh'][0],simplify=0.97,texture_size=2048)
        glb_maker.export("output.glb")
        
        with open("output.glb", "rb") as f:
            glb_data = f.read()
        
        # Clean up
        if os.path.exists("output.glb"):
            os.remove("output.glb")
            
        return glb_data
        
    except Exception as e:
        print(f"Error in 3D generation: {str(e)}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path}")
        raise

@stub.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    memory=8192  # Less memory for simpler model
)
def generate_image_simple(prompt: str) -> bytes:
    """Fallback image generation with SD 1.5 for lower memory usage"""
    import torch
    from diffusers import StableDiffusionPipeline
    import gc
    from pathlib import Path
    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    pipe = None
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        device = "cuda"
        torch_dtype = torch.float16


        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        pipe.enable_sequential_cpu_offload()
        
        # Generate image
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                width=512,
                height=512
            ).images[0]

        # Save image
        output_path = Path("/output_simple.png")
        image.save(output_path)

        with open(output_path, "rb") as f:
            image_bytes = f.read()

        return image_bytes

    except Exception as e:
        print(f"Error in simple image generation: {str(e)}")
        raise
    finally:
        if pipe is not None:
            del pipe
        torch.cuda.empty_cache()
        gc.collect()

@stub.local_entrypoint()
def main():
    from pathlib import Path

    prompt = "A detailed, realistic, isolated modern space rocket on a plain white background, no shadows or reflections, fully visible from tip to engine, centered in frame, sleek metallic textures with visible panel lines and rivets, minimal stylization, professional lighting, sharp focus, high resolution, no background, no text,main rocket taller than other boosters"
    print("🚀 Generating image...")

    try:
        # Try SDXL first
        print("Attempting SDXL generation...")
        try:
            image_bytes = generate_image.remote(prompt)
            print("✅ SDXL generation successful")
        except Exception as e:
            print(f"❌ SDXL failed: {str(e)}")
            print("🔄 Falling back to SD 1.5...")
            image_bytes = generate_image_simple.remote(prompt)
            print("✅ SD 1.5 generation successful")

        # Create local output directory
        output_dir = Path("interim_output")
        output_dir.mkdir(exist_ok=True)

        # Save the generated image locally
        local_image_path = output_dir / "spaceship.png"
        with open(local_image_path, "wb") as f:
            f.write(image_bytes)

        print(f"✅ Image saved to {local_image_path}")

        # Generate 3D model using the saved image
        print("🛠️ Generating 3D model...")
        mesh_bytes = generate_3dModel.remote(image_bytes)

        # Save the 3D model locally
        local_model_path = output_dir / "spaceship.glb"
        with open(local_model_path, "wb") as f:
            f.write(mesh_bytes)

        print(f"✅ 3D model saved to {local_model_path}")

    except Exception as e:
        print(f"Failed to generate or save image/model: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()