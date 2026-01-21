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
    Returns a list of (name, x, y) for 4 overlapping tiles.
    Target Size: 1920
    Tile Size: 1024
    """
    size = conf.SR_TARGET_SIZE
    tile = conf.SR_TILE_SIZE
    
    # Coords are (left, top)
    # TL: 0, 0
    # TR: 1920 - 1024, 0
    # BL: 0, 1920 - 1024
    # BR: 1920 - 1024, 1920 - 1024
    
    offset = size - tile # 896
    
    coords = [
        ("TL", 0, 0),
        ("TR", offset, 0),
        ("BL", 0, offset),
        ("BR", offset, offset)
    ]
    return coords

def create_gradient_mask(tile_size, overlap):
    """
    Creates a weight mask for blending. 
    Simple approach: 1 in the center, linear fade to 0 at edges.
    However, for strict 4-tile grid, we can just blend the overlapping strips.
    
    To implement a robust weighted merge, we create a weight map for the whole 1920x1920 canvas.
    """
    # Create a 2D gaussian-like or trapezoidal mask for a single tile
    # For this specific 4-tile layout, a simple "feathered edge" mask is sufficient.
    
    # Let's use a standard trapezoid fade (keeps center 100% original, fades edges)
    # 0 -> overlap : fade 0 to 1
    # overlap -> size - overlap : keep 1
    # size - overlap -> size : fade 1 to 0
    
    x = np.linspace(0, 1, tile_size)
    y = np.linspace(0, 1, tile_size)
    
    # Define fade function
    def fade(arr):
        # Fade in first 'overlap' pixels
        fade_len = overlap
        result = np.ones_like(arr)
        
        # Left/Top edge fade
        result[:fade_len] = np.linspace(0, 1, fade_len)
        # Right/Bottom edge fade
        result[-fade_len:] = np.linspace(1, 0, fade_len)
        return result

    mask_x = fade(np.arange(tile_size))
    mask_y = fade(np.arange(tile_size))
    
    # Outer product to make 2D mask
    mask_2d = np.outer(mask_y, mask_x)
    return mask_2d

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
    
    # Prepare canvas for weighted sum
    # Shape: (H, W, 3)
    canvas = np.zeros((conf.SR_TARGET_SIZE, conf.SR_TARGET_SIZE, 3), dtype=np.float32)
    weight_map = np.zeros((conf.SR_TARGET_SIZE, conf.SR_TARGET_SIZE), dtype=np.float32)
    
    tile_mask = create_gradient_mask(conf.SR_TILE_SIZE, conf.SR_OVERLAP)
    # Expand mask to 3 channels for image multiplication: (H, W, 1)
    tile_mask_3d = tile_mask[:, :, np.newaxis]

    # Ensure pipe is in Img2Img mode
    if not isinstance(pipe, AutoPipelineForImage2Image):
        pipe = AutoPipelineForImage2Image.from_pipe(pipe)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device).manual_seed(42) # Fixed seed for consistency across tiles

    # 3. Process Tiles
    print("   |-- [2/4] Processing Tiles (Img2Img)...")
    for name, x, y in coords:
        print(f"       > Tile {name} at ({x}, {y})...")
        
        # Crop
        box = (x, y, x + conf.SR_TILE_SIZE, y + conf.SR_TILE_SIZE)
        tile_img = upscaled_img.crop(box)
        
        # Img2Img Refinement
        refined_tile = pipe(
            prompt=pt.PROMPT_SR_TEXT,
            negative_prompt=pt.NEGATIVE_PROMPT_TEXT,
            image=tile_img,
            strength=conf.SR_STRENGTH,
            guidance_scale=conf.SR_GUIDANCE_SCALE,
            num_inference_steps=conf.SR_INFERENCE_STEPS,
            target_size=(conf.SR_TILE_SIZE, conf.SR_TILE_SIZE), # 1024
            original_size=conf.ORIGINAL_SIZE, # High res assumption
            negative_original_size=conf.NEGATIVE_ORIGINAL_SIZE,
            generator=generator,
            output_type="pil"
        ).images[0]
        
        # Convert to numpy
        refined_np = np.array(refined_tile).astype(np.float32)
        
        # Add to canvas (weighted)
        canvas[y:y+conf.SR_TILE_SIZE, x:x+conf.SR_TILE_SIZE] += refined_np * tile_mask_3d
        weight_map[y:y+conf.SR_TILE_SIZE, x:x+conf.SR_TILE_SIZE] += tile_mask
        
    # 4. Merge
    print("   |-- [3/4] Merging Tiles...")
    # Avoid division by zero
    weight_map = np.maximum(weight_map, 1e-5)
    weight_map_3d = weight_map[:, :, np.newaxis]
    
    final_np = canvas / weight_map_3d
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

    # Load Model (Reusing logic from t2i)
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