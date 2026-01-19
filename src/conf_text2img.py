# src/conf_text2img.py

import os

##### Section I : Path Configuration #####

# Step 1 : Define base paths
# Assuming the script is run from the project root directory
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "mod")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output/txt2img")

# Step 2 : Define specific file paths
# Model filename
MODEL_FILENAME = "hardcoreAsianCosplay_ilV11.safetensors"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

##### Section II : Generation Configuration #####

# Step 1 : Output settings
BASE_FILENAME_PREFIX = "cos"
NUM_IMAGES_TO_GENERATE = 10

# Step 2 : Image dimensions
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

# --- Stage 1: Base Generation (Structure) ---
# High CFG to force the Cosplay outfit accuracy
BASE_INFERENCE_STEPS = 30
BASE_GUIDANCE_SCALE = 7.0

# --- Stage 2: Refinement (Texture & Realism) ---
# Low CFG to allow natural skin texture
# Strength determines how many steps we "go back". 
# 0.3 means roughly last 30% of steps are re-done with these new settings.
REFINE_INFERENCE_STEPS = 50
REFINE_GUIDANCE_SCALE = 4.0
REFINE_STRENGTH = 0.4

# --- SDXL Specifics ---
TARGET_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
ORIGINAL_SIZE = (IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2)
NEGATIVE_ORIGINAL_SIZE = (512, 512)