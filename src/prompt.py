# src/prompt.py

##### Section I : Text to Image Prompt #####

# Step 1 : Positive
PROMPT_TEXT = (
# Camera 
    "fujifilm xt4, film grain, high iso, raw style, gravure, "
# Lighting
    "chiaroscuro, hard shadows, dimly lit, "
# Actor
    # Basic
    "Japanese cosplayer girl, "
    # Detail
    "Raiden Shogun Cosplay, purple braided hair, kimono, cleavage, "
    # Makeup
    "soft makeup, realistic eyes, "
    "sweat, mole, "
    # Action
    "looking at viewer, selfie, "
    # Background
    "bedroom, "
# Real Config
    "textured skin, visible pores, "
)

# Step 2 : Negative
NEGATIVE_PROMPT_TEXT = (
# CG & Artificial
    "anime, cartoon, 3d, 3d render, cgm, illustration, painting, drawing, art, "
    "doll, plastic, wax figure, action figure, "
# Lighting
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