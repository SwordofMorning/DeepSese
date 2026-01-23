# src/t2i/t2i.py

import torch
from diffusers import StableDiffusionXLPipeline, AutoPipelineForText2Image, AutoPipelineForImage2Image, EulerAncestralDiscreteScheduler
import os
import sys

# Adjusted imports for new structure
from src.conf import conf
from src.conf import prompt as pt

##### Section I : Core Logic #####

def load_initial_pipeline(model_path):
    print(f"[INFO] Loading SDXL Model from: {model_path} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    try:
        # Loading single files (.safetensors)
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
        )
        
        # Scheduler
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        # --- Optimizations for 8GB VRAM (RTX 4060) ---
        pipe.enable_model_cpu_offload() 
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        
        print("[INFO] Model loaded successfully with VRAM optimizations!")
        return pipe
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def process_two_stage_generation(pipe, index, total_images):
    print(f"\n[INFO] Processing Task {index + 1}/{total_images} ...")
    
    # Generate a random seed
    seed = torch.randint(0, 2**32, (1,)).item()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(seed)
    
    # Stage 1: Text to Image
    print(f"   |-- [Stage 1] Generating Base Structure (CFG: {conf.BASE_GUIDANCE_SCALE})...")
    
    if not isinstance(pipe, AutoPipelineForText2Image):
        pipe = AutoPipelineForText2Image.from_pipe(pipe)
    
    try:
        base_image = pipe(
            prompt=pt.PROMPT_TEXT,
            negative_prompt=pt.NEGATIVE_PROMPT_TEXT,
            height=conf.IMAGE_HEIGHT, 
            width=conf.IMAGE_WIDTH,   
            guidance_scale=conf.BASE_GUIDANCE_SCALE, 
            num_inference_steps=conf.BASE_INFERENCE_STEPS, 
            target_size=conf.TARGET_SIZE,
            original_size=conf.ORIGINAL_SIZE, 
            negative_original_size=conf.NEGATIVE_ORIGINAL_SIZE,
            generator=generator,
            output_type="pil"
        ).images[0]
    except Exception as e:
        print(f"[ERROR] Stage 1 failed: {e}")
        return None, None, None, pipe

    # Stage 2: Refinement
    print(f"   |-- [Stage 2] Refining Texture (CFG: {conf.REFINE_GUIDANCE_SCALE}, Str: {conf.REFINE_STRENGTH})...")
    
    pipe = AutoPipelineForImage2Image.from_pipe(pipe)
    
    try:
        refined_image = pipe(
            prompt=pt.PROMPT_REFINE_TEXT,
            negative_prompt=pt.NEGATIVE_PROMPT_TEXT,
            image=base_image,
            strength=conf.REFINE_STRENGTH,
            guidance_scale=conf.REFINE_GUIDANCE_SCALE,
            num_inference_steps=conf.REFINE_INFERENCE_STEPS,
            target_size=conf.TARGET_SIZE,
            original_size=conf.ORIGINAL_SIZE, 
            negative_original_size=conf.NEGATIVE_ORIGINAL_SIZE,
            generator=generator
        ).images[0]
        
        return base_image, refined_image, seed, pipe
    except Exception as e:
        print(f"[ERROR] Stage 2 failed: {e}")
        return base_image, None, seed, pipe

##### Section II : Module Execution Entry #####

def run_task(num_images=None):
    """
    Entry point for the T2I module.
    """
    # Determine number of images
    count = num_images if num_images is not None else conf.NUM_IMAGES_TO_GENERATE
    
    if not os.path.exists(conf.MODEL_PATH):
        print(f"[ERROR] Model file not found: {conf.MODEL_PATH}")
        return

    if not os.path.exists(conf.OUTPUT_DIR_T2I):
        os.makedirs(conf.OUTPUT_DIR_T2I)

    # Load Initial Pipeline
    pipe = load_initial_pipeline(conf.MODEL_PATH)

    print("========================================")
    print(f"Batch Task: {count} images (Two-Stage Optimized)")
    print("========================================")

    for i in range(count):
        # Pass 0 as index for now or track manually
        base_img, final_img, seed, pipe = process_two_stage_generation(pipe, i, count)

        if final_img:
            filename = f"{conf.BASE_FILENAME_PREFIX}_{i+1:02d}_final.png"
            save_path = os.path.join(conf.OUTPUT_DIR_T2I, filename)
            final_img.save(save_path)
            print(f"[SUCCESS] Saved: {filename} (Seed: {seed})")
        else:
            print(f"[SKIP] Failed to generate image {i+1}")

    print("========================================")
    print("T2I tasks completed!")