# src/prompt.py

# --- Note ---
# The order of the Prompts affects the output of SD (Simultaneous Visual Representation).
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
    "Japanese cosplayer girl , "
    # Detail
    "Raiden Shogun Cosplay, kimono, cleavage, "
    # Makeup
    "soft makeup, realistic eyes, "
    "sweat, mole, "
    # Action
    "looking at viewer, selfie angle, flash photography, "
    # Background
    "bedroom, messy bed sheet,"
# Skin
    "natural skin texture, visible pores, slight skin imperfections, "
# Camera
    "fujifilm xt4, film grain, high iso, raw style, gravure"
# Light
    "chiaroscuro, hard shadows, dimly lit, "
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
    "heavy makeup, clown makeup, "
# Anatomy
    "bad anatomy, bad hands, missing fingers, extra digits, "
# Quality
    "blurry, low quality, text, watermark"
)