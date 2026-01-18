# src/prompt.py

# --- Note ---
# The order of the Prompts affects the output of SD.
#
# For example: 
#
# Moving the Camera section forward, will result in a more professional-looking output; 
# moving the Light section forward, will emphasize lighting and shadow effects.
# --- Note ---

##### Section I : Text to Image Prompt #####

# Step 1 : Positive
PROMPT_TEXT = (
# Actor
    # Basic
    "Young Japanese cosplayer girl , "
    "soft cheeks, "
    # Detail
    "Raiden Shogun Cosplay, "
    "kimono, cleavage, "
    # Makeup
    "soft makeup, realistic eyes, "
    "mole," # as noise
    # Action
    "looking at viewer, selfie angle, flash photography, "
    # Background
    "bedroom,"
# Skin
    "natural skin texture, visible pores, slight skin imperfections, "
# Camera
    "fujifilm xt4, film grain, high iso, raw style, gravure"
# Light
    "chiaroscuro, hard shadows, "
)

# Step 2 : Negative
NEGATIVE_PROMPT_TEXT = (
# CG & Artificial
    "anime, cartoon, 3d, 3d render, cgm, illustration, painting, drawing, art, "
    "doll, plastic, wax figure, action figure, "
# Light
    "flat lighting, soft lighting, even lighting, bright, overexposed, studio lighting, "
# Skin flaws
    "perfect skin, airbrushed, photoshop, retouched, smooth skin, oil skin, "
# Makeup
    "heavy makeup, clown makeup, heavy eyeshadow, "
# Anatomy
    "bad anatomy, bad hands, missing fingers, extra digits, "
# Face Structure
    "long face, sharp chin, pointed chin, sharp jawline, "
    "gaunt, sunken cheeks, mature face, "
    "k-pop style, plastic surgery, "
# Quality
    "blurry, low quality, text, watermark"
)