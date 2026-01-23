# src/conf/conf.py

import os

##### Section I : Path Configuration #####

# Step 1 : Define base paths
# We assume the script is run from Project Root
ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "mod")
# T2I Output
OUTPUT_DIR_T2I = os.path.join(ROOT_DIR, "output/txt2img")
# SR Output
OUTPUT_DIR_SR = os.path.join(ROOT_DIR, "output/sr")

# Step 2 : Define specific file paths
MODEL_FILENAME = "hardcoreAsianCosplay_ilV11.safetensors"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

##### Section II : T2I Configuration #####

# Image dimensions
IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 1024

# T2I - Stage 1
BASE_FILENAME_PREFIX = "cos"
NUM_IMAGES_TO_GENERATE = 10
BASE_INFERENCE_STEPS = 30
BASE_GUIDANCE_SCALE = 7.0

# T2I - Stage 2 (Refinement)
REFINE_INFERENCE_STEPS = 50
REFINE_GUIDANCE_SCALE = 4.0
REFINE_STRENGTH = 0.4

# SDXL Specifics
TARGET_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
ORIGINAL_SIZE = (IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2)
NEGATIVE_ORIGINAL_SIZE = (512, 512)

##### Section III : SR Configuration #####

# Step 1 : Upscaling Dimensions
# 1024 -> 1920 (approx 1.875x)
SR_TARGET_SIZE = 1920
SR_TILE_SIZE = 1024
# Overlap is calculated: (1024 * 2) - 1920 = 2048 - 1920 = 128 pixels
SR_OVERLAP = 128

# Step 2 : Refinement Parameters (Per Tile)
# We use a lower strength to preserve the original structure while adding details
SR_STRENGTH = 0.35
SR_GUIDANCE_SCALE = 5.0
SR_INFERENCE_STEPS = 40