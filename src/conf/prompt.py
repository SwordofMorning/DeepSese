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
    # "soft cheeks, "
    # Detail
    "Raiden Shogun Cosplay, "
    "kimono, cleavage, "
    # Makeup
    "soft makeup, realistic eyes, "
    "mole, sweat, " # as noise
    # Action
    "looking at viewer, selfie angle, flash photography, "
    # Background
    "bedroom, messy sheets, "
# Skin
    "natural skin, visible pores, slight skin imperfections, "
# Camera
    "fujifilm xt4, film grain, high iso, raw style, gravure"
# Light
    "chiaroscuro, hard shadows"
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

##### Section II : Image to Image Prompt (Refiner) #####

# In the refinement stage, we emphasize texture keywords.
# Usually we can reuse the PROMPT_TEXT, but explicit definition helps flexibility.
PROMPT_REFINE_TEXT = PROMPT_TEXT

##### Section III : Super-Resolution Prompt #####

# For SR, we emphasize texture even more to ensure the upscale adds detail
# rather than smoothing things out.
PROMPT_SR_TEXT = (
# Actor
    # Basic
    "Young Japanese cosplayer girl , "
    # Detail
    "Raiden Shogun Cosplay, "
    "kimono, cleavage, "
    # Makeup
    "soft makeup, realistic eyes, "
    # Background
    "bedroom, messy sheets, "
# Skin
    "natural skin texture, visible pores, slight skin imperfections, "
# Camera
    "fujifilm xt4, film grain, high iso, raw style, gravure"
# Light
    "chiaroscuro, hard shadows"
)