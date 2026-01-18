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
# Number of denoising steps
INFERENCE_STEPS = 40
# CFG Scale (Guidance Scale)
GUIDANCE_SCALE = 6.0

# Step 4 : SDXL Specifics
TARGET_SIZE = (1024, 1024)
ORIGINAL_SIZE = (1024, 1024)
NEGATIVE_ORIGINAL_SIZE = (512, 512)