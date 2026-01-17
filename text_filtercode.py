import os

# PATHS (do NOT change unless your folder names are different)
IMAGES_DIR = "dataset/images/images_300"
CAPTIONS_FILE = "dataset/Text/Flickr8k_text/caption.txt"
OUTPUT_FILE = "dataset/Text/Flickr8k_text/filtered_text.txt"

# ----------------------------------------------------
# STEP 1: Collect image base names from images_300
# ----------------------------------------------------
image_bases = set()

for img in os.listdir(IMAGES_DIR):
    if img.lower().endswith(".jpg"):
        base_name = os.path.splitext(img)[0].lower()
        image_bases.add(base_name)

print("Images found in images_300:", len(image_bases))
print("Sample images:", list(image_bases)[:5])

# ----------------------------------------------------
# STEP 2: Filter captions (1 caption per image)
# Flickr8k format:
# image.jpg#0 A caption sentence
# ----------------------------------------------------
filtered_captions = {}

with open(CAPTIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        # Split at FIRST space
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue

        image_id, caption = parts

        # Remove #0, #1 etc and .jpg
        image_name = image_id.split("#")[0]
        image_base = os.path.splitext(image_name)[0].lower()

        # Keep only one caption per image
        if image_base in image_bases and image_base not in filtered_captions:
            filtered_captions[image_base] = caption

# ----------------------------------------------------
# STEP 3: Save filtered captions
# ----------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for img, cap in filtered_captions.items():
        f.write(f"{img}.jpg\t{cap}\n")

print(f"Done! {len(filtered_captions)} captions saved")
