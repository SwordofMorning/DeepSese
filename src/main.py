# src/main.py

import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import os
import sys

# Import local modules
# Ensure the current directory is in sys.path to allow imports if run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import conf_text2img as conf
import prompt as pt

##### Section I : Core Logic #####

def load_pipeline(model_path):
    # Step 1 : Log start
    print(f"[INFO] Loading SDXL Model from: {model_path} ...")
    
    # Step 2 : Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")
    
    try:
        # Step 3 : Initialize Pipeline
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True,
        )
        
        # Step 4 : Configure Scheduler
        
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        #     pipe.scheduler.config,
        #     use_karras_sigmas=True,
        #     algorithm_type="dpmsolver++"
        # )

        # Note: Euler a seems better.
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )

        # Step 5 : Optimizations
        pipe.enable_model_cpu_offload() 
        pipe.enable_vae_slicing()
        
        print("[INFO] SDXL Model loaded successfully! (Scheduler: Euler a)")
        return pipe
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def generate_image(pipe, prompt_text, negative_prompt_text, index):
    # Step 1 : Log generation start
    print(f"[INFO] Generating image {index + 1}/{conf.NUM_IMAGES_TO_GENERATE} ...")
    
    try:
        # Step 2 : Generate random seed
        seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Step 3 : Run inference
        image = pipe(
            prompt=prompt_text,
            negative_prompt=negative_prompt_text,
            height=conf.IMAGE_HEIGHT, 
            width=conf.IMAGE_WIDTH,   
            guidance_scale=conf.GUIDANCE_SCALE, 
            num_inference_steps=conf.INFERENCE_STEPS, 
            target_size=conf.TARGET_SIZE,
            original_size=conf.ORIGINAL_SIZE, 
            negative_original_size=conf.NEGATIVE_ORIGINAL_SIZE,
            generator=generator
        ).images[0]
        
        return image, seed
    except Exception as e:
        print(f"[ERROR] Failed to generate image {index + 1}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

##### Section II : Main Execution #####

def main():
    # Step 1 : Validate Model Path
    if not os.path.exists(conf.MODEL_PATH):
        print(f"[ERROR] Model file not found: {conf.MODEL_PATH}")
        return

    # Step 2 : Create Output Directory
    if not os.path.exists(conf.OUTPUT_DIR):
        os.makedirs(conf.OUTPUT_DIR)
        print(f"[INFO] Created output directory: {conf.OUTPUT_DIR}")

    # Step 3 : Load Model
    pipe = load_pipeline(conf.MODEL_PATH)

    print("========================================")
    print(f"Batch Task: {conf.NUM_IMAGES_TO_GENERATE} images")
    print("========================================")

    # Step 4 : Generation Loop
    for i in range(conf.NUM_IMAGES_TO_GENERATE):
        result_image, seed = generate_image(pipe, pt.PROMPT_TEXT, pt.NEGATIVE_PROMPT_TEXT, i)

        # Step 5 : Save Result
        if result_image:
            filename = f"{conf.BASE_FILENAME_PREFIX}_{i+1:02d}.png"
            save_path = os.path.join(conf.OUTPUT_DIR, filename)
            
            result_image.save(save_path)
            print(f"[SUCCESS] Saved: {save_path} (Seed: {seed})")
        else:
            print(f"[SKIP] Skipped image {i+1}")

    print("========================================")
    print("All tasks completed!")

if __name__ == "__main__":
    main()