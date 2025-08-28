import os

from PIL import Image

input_folder = "/Users/jazav7774/Data/Mammo/images_png"
output_folder = "/Users/jazav7774/Data/Mammo/images_png_396"
os.makedirs(output_folder, exist_ok=True)

target_size = (660, 396)

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(input_folder, filename))
        img = img.resize(
            target_size, Image.LANCZOS
        )  # or Image.LANCZOS for better quality
        img.save(os.path.join(output_folder, filename))
