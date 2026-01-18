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

# Step 3 : Inference settings
# Steps of "noise reduction", reduced steps to prevent "plastic" skin smoothing
INFERENCE_STEPS = 40
# Prompt relevance: Reduced CFG to allow more natural variation and texture, 4.0 to 6.0
GUIDANCE_SCALE = 4.2

# Step 4 : SDXL Specifics
TARGET_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
ORIGINAL_SIZE = (IMAGE_HEIGHT * 2, IMAGE_WIDTH * 2)
NEGATIVE_ORIGINAL_SIZE = (512, 512)