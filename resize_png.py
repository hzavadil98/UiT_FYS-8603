import os

from PIL import Image
from tqdm import tqdm

input_folder = "/Users/jazav7774/Data/Mammo/images_png"
output_folder = "/Users/jazav7774/Data/Mammo/images_png_396"
os.makedirs(output_folder, exist_ok=True)

target_size = (369, 660)

for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(input_folder, filename))
        img = img.resize(
            target_size, Image.LANCZOS
        )  # or Image.LANCZOS for better quality
        img.save(os.path.join(output_folder, filename))
