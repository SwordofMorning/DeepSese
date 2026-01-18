# src/prompt.py

##### Section I : Text to Image Prompt #####

# Step 1 : Positive
PROMPT_TEXT = (
# Actor
    # Basic
    "Japanese cosplayer girl , "
    # Detail
    "Raiden Shogun Cosplay, purple braided hair, kimono, cleavage, "
    # Makeup
    "soft makeup, realistic eyes, "
    "sweat, mole, makeup, "
    # Action
    "looking at viewer, selfie angle, flash photography, "
    # Background
    "bedroom, messy bed sheet,"
# Real Config
    "natural skin texture, visible pores, slight skin imperfections, "
# Camera Config
    "fujifilm xt4, film grain, high iso, raw style, gravure"
)

# Step 2 : Negative
NEGATIVE_PROMPT_TEXT = (
    "anime, cartoon, 3d, 3d render, cgm, illustration, painting, drawing, art, "
    "doll, plastic, wax figure, action figure, "
    "perfect skin, airbrushed, photoshop, retouched, smooth skin, oil skin, "
    "makeup, heavy makeup, "
    "bad anatomy, bad hands, missing fingers, extra digits, "
    "blurry, low quality, text, watermark"
)