# src/main.py

import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, EulerAncestralDiscreteScheduler
import os
import sys

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import conf_text2img as conf
import prompt as pt

##### Section I : Core Logic #####

def load_base_pipeline(model_path):
    print(f"[INFO] Loading SDXL Base Model from: {model_path} ...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load the initial Text-to-Image pipeline
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
        )
        
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        pipe.enable_model_cpu_offload() 
        pipe.enable_vae_slicing()
        
        print("[INFO] Base Pipeline loaded successfully!")
        return pipe
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

def get_refiner_pipeline(base_pipe):
    # Create an Image-to-Image pipeline sharing the same components (UNet, VAE)
    # This uses zero additional VRAM for model weights.
    print("[INFO] creating Refiner Pipeline (Img2Img) from Base components...")
    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pipe(base_pipe)
    return refiner_pipe

def process_image_pipeline(base_pipe, refiner_pipe, index):
    print(f"\n[INFO] Processing Task {index + 1}/{conf.NUM_IMAGES_TO_GENERATE} ...")
    
    # Common Generator for reproducibility
    seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # --- Stage 1: Text to Image (Structure) ---
    print(f"   |-- [Stage 1] Generating Base Structure (CFG: {conf.BASE_GUIDANCE_SCALE})...")
    try:
        base_image = base_pipe(
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
        return None, None, None

    # --- Stage 2: Image to Image (Refinement) ---
    print(f"   |-- [Stage 2] Refining Texture (CFG: {conf.REFINE_GUIDANCE_SCALE}, Strength: {conf.REFINE_STRENGTH})...")
    try:
        # We reuse the same generator to maintain noise coherence, 
        # or create a new one for variation. Reusing is usually safer for refinement.
        refined_image = refiner_pipe(
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
        
        return base_image, refined_image, seed
    except Exception as e:
        print(f"[ERROR] Stage 2 failed: {e}")
        return base_image, None, seed

##### Section II : Main Execution #####

def main():
    if not os.path.exists(conf.MODEL_PATH):
        print(f"[ERROR] Model file not found: {conf.MODEL_PATH}")
        return

    if not os.path.exists(conf.OUTPUT_DIR):
        os.makedirs(conf.OUTPUT_DIR)

    # 1. Load Base Pipe
    base_pipe = load_base_pipeline(conf.MODEL_PATH)
    
    # 2. Create Refiner Pipe (Shares memory)
    refiner_pipe = get_refiner_pipeline(base_pipe)

    print("========================================")
    print(f"Batch Task: {conf.NUM_IMAGES_TO_GENERATE} images (Two-Stage)")
    print("========================================")

    for i in range(conf.NUM_IMAGES_TO_GENERATE):
        base_img, final_img, seed = process_image_pipeline(base_pipe, refiner_pipe, i)

        if final_img:
            # Save Final
            filename = f"{conf.BASE_FILENAME_PREFIX}_{i+1:02d}_final.png"
            save_path = os.path.join(conf.OUTPUT_DIR, filename)
            final_img.save(save_path)
            
            # Optional: Save Base for comparison
            # base_filename = f"{conf.BASE_FILENAME_PREFIX}_{i+1:02d}_base.png"
            # base_img.save(os.path.join(conf.OUTPUT_DIR, base_filename))
            
            print(f"[SUCCESS] Saved: {filename} (Seed: {seed})")
        else:
            print(f"[SKIP] Failed to generate image {i+1}")

    print("========================================")
    print("All tasks completed!")

if __name__ == "__main__":
    main()