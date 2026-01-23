# src/sr/sr.py

import os
import sys
import numpy as np
from PIL import Image
import torch
from diffusers import AutoPipelineForImage2Image

# Import shared config and t2i pipeline loader
from src.conf import conf
from src.conf import prompt as pt
from src.t2i import t2i

##### Section I : Helper Logic (Lanczos & Tiling) #####

def upscale_lanczos(image, target_size):
    """Upscales image using Lanczos resampling."""
    return image.resize((target_size, target_size), Image.LANCZOS)

def get_tile_coordinates():
    """
    Returns a list of (name, x, y, fade_sides) for 4 overlapping tiles.
    fade_sides order: [Top, Bottom, Left, Right] - Boolean
    """
    size = conf.SR_TARGET_SIZE
    tile = conf.SR_TILE_SIZE
    offset = size - tile # 896
    
    # Define which sides should act as borders (overlap) and need fading.
    # True = Fade this side (Internal edge)
    # False = Keep opaque (External edge)
    
    coords = [
        # TL: Fade Bottom, Right
        ("TL", 0, 0, {"top": False, "bottom": True, "left": False, "right": True}),
        
        # TR: Fade Bottom, Left
        ("TR", offset, 0, {"top": False, "bottom": True, "left": True, "right": False}),
        
        # BL: Fade Top, Right
        ("BL", 0, offset, {"top": True, "bottom": False, "left": False, "right": True}),
        
        # BR: Fade Top, Left
        ("BR", offset, offset, {"top": True, "bottom": False, "left": True, "right": False})
    ]
    return coords

def create_tile_mask(tile_size, overlap, sides):
    """
    Creates a smart weight mask.
    sides: dict {"top": bool, "bottom": bool, ...}
    """
    # Start with a mask of all 1.0
    mask = np.ones((tile_size, tile_size), dtype=np.float32)
    
    # Generate a linear gradient 0 -> 1
    # We use explicit indices to ensure robust fading
    fade_ramp = np.linspace(0, 1, overlap)
    
    # Apply fade to specific sides if requested
    
    if sides["top"]:
        # Fade the top rows (0 to overlap) from 0 to 1
        # We multiply the existing values to support corners (where two fades meet)
        for i in range(overlap):
            mask[i, :] *= fade_ramp[i]
            
    if sides["bottom"]:
        # Fade the bottom rows from 1 to 0
        for i in range(overlap):
            mask[tile_size - 1 - i, :] *= fade_ramp[i]
            
    if sides["left"]:
        # Fade the left columns from 0 to 1
        for i in range(overlap):
            mask[:, i] *= fade_ramp[i]
            
    if sides["right"]:
        # Fade the right columns from 1 to 0
        for i in range(overlap):
            mask[:, tile_size - 1 - i] *= fade_ramp[i]

    # Add an extra channel dimension for broadcasting: (H, W, 1)
    return mask[:, :, np.newaxis]

##### Section II : Core SR Logic #####

def process_single_image_sr(pipe, image_path, output_dir):
    filename = os.path.basename(image_path)
    print(f"\n[SR] Processing: {filename}")
    
    try:
        original_img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[ERROR] Could not open image {image_path}: {e}")
        return

    # 1. Pre-upscale
    print("   |-- [1/4] Lanczos Upscaling to 1920x1920...")
    upscaled_img = upscale_lanczos(original_img, conf.SR_TARGET_SIZE)
    
    # 2. Prepare for Tiling
    coords = get_tile_coordinates()
    
    # Prepare canvas
    canvas = np.zeros((conf.SR_TARGET_SIZE, conf.SR_TARGET_SIZE, 3), dtype=np.float32)
    weight_map = np.zeros((conf.SR_TARGET_SIZE, conf.SR_TARGET_SIZE, 1), dtype=np.float32)
    
    # Ensure pipe is in Img2Img mode
    if not isinstance(pipe, AutoPipelineForImage2Image):
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)

    # Auto-detect device (Fix for CI/CD compatibility)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(42)

    # 3. Process Tiles
    print("   |-- [2/4] Processing Tiles (Img2Img)...")
    for name, x, y, sides in coords:
        print(f"       > Tile {name} at ({x}, {y})...")
        
        # Crop
        box = (x, y, x + conf.SR_TILE_SIZE, y + conf.SR_TILE_SIZE)
        tile_img = upscaled_img.crop(box)
        
        # Generate Mask for this specific tile position
        tile_mask_3d = create_tile_mask(conf.SR_TILE_SIZE, conf.SR_OVERLAP, sides)
        
        # Img2Img Refinement
        # [FIX] We Force original_size to match target_size (1024)
        # This prevents SDXL from shrinking the content/adding black borders
        refined_tile = pipe(
            prompt=pt.PROMPT_SR_TEXT,
            negative_prompt=pt.NEGATIVE_PROMPT_TEXT,
            image=tile_img,
            strength=conf.SR_STRENGTH,
            guidance_scale=conf.SR_GUIDANCE_SCALE,
            num_inference_steps=conf.SR_INFERENCE_STEPS,
            target_size=(conf.SR_TILE_SIZE, conf.SR_TILE_SIZE), 
            original_size=(conf.SR_TILE_SIZE, conf.SR_TILE_SIZE), # FIX: Do not use 2048 here
            negative_original_size=conf.NEGATIVE_ORIGINAL_SIZE,
            generator=generator,
            output_type="pil"
        ).images[0]
        
        refined_np = np.array(refined_tile).astype(np.float32)
        
        # Add to canvas
        # Note: tile_mask_3d is (1024, 1024, 1), implicit broadcast works for refined_np (1024, 1024, 3)
        canvas[y:y+conf.SR_TILE_SIZE, x:x+conf.SR_TILE_SIZE] += refined_np * tile_mask_3d
        
        # Accumulate weights (squeeze the last dim for weight map if needed, or keep 3d)
        # Here we keep weight_map as (H, W, 1) to divide easily
        weight_map[y:y+conf.SR_TILE_SIZE, x:x+conf.SR_TILE_SIZE] += tile_mask_3d
        
    # 4. Merge
    print("   |-- [3/4] Merging Tiles...")
    # Normalize
    weight_map = np.maximum(weight_map, 1e-5)
    final_np = canvas / weight_map
    
    final_np = np.clip(final_np, 0, 255).astype(np.uint8)
    final_img = Image.fromarray(final_np)
    
    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_name = f"SR_{filename}"
    save_path = os.path.join(output_dir, save_name)
    final_img.save(save_path)
    print(f"   |-- [4/4] Saved: {save_path}")

##### Section III : Module Entry #####

def run_task(file_path=None, folder_path=None):
    if not file_path and not folder_path:
        print("[ERROR] SR Task requires --file or --folder argument.")
        return

    # Load Model
    pipe = t2i.load_initial_pipeline(conf.MODEL_PATH)
    
    targets = []
    
    if file_path:
        if os.path.isfile(file_path):
            targets.append(file_path)
        else:
            print(f"[ERROR] File not found: {file_path}")
            
    if folder_path:
        if os.path.isdir(folder_path):
            valid_exts = ('.png', '.jpg', '.jpeg')
            for f in os.listdir(folder_path):
                if f.lower().endswith(valid_exts) and "SR_" not in f:
                    targets.append(os.path.join(folder_path, f))
        else:
            print(f"[ERROR] Folder not found: {folder_path}")
            
    if not targets:
        print("[WARN] No images found to process.")
        return

    print("========================================")
    print(f"SR Task: {len(targets)} images")
    print("========================================")

    for img_path in targets:
        process_single_image_sr(pipe, img_path, conf.OUTPUT_DIR_SR)
        
    print("========================================")
    print("SR tasks completed!")