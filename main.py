import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
from patternModel import ImageConditioning
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_encoder_path = "./models/image_encoder/"
checkpoint = "./models/model_checkpoint.bin"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on: ", device)
base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"

def resize_without_padding(image_pil, target_size=(900, 1200)):
    # Convert PIL Image to NumPy array
    image = np.array(image_pil)
    # Resize the image without preserving aspect ratio (stretching)
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
    return resized_image

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)


print("Checking Image Encoder path:", image_encoder_path)
print("Exists:", os.path.exists(image_encoder_path))

print("Checking Checkpoint path:", checkpoint)
print("Exists:", os.path.exists(checkpoint))

# state_dict = torch.load(checkpoint, map_location="cpu")
# state_dict.keys()

conditioning_model = ImageConditioning(pipe, image_encoder_path,checkpoint, device)

# Exporting output for all test sets ---------------------------------------------------------------------

# input_folder = '/content/test'
# output_folder = '/content/output'

# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)

# for i in range(50):
#     # Construct the input file path
#     input_file_path_png = os.path.join(input_folder, f"{i}.png")
#     input_file_path_jpg = os.path.join(input_folder, f"{i}.jpg")

#     # Check if the file exists (either png or jpg)
#     if os.path.exists(input_file_path_png):
#         input_file_path = input_file_path_png
#     elif os.path.exists(input_file_path_jpg):
#         input_file_path = input_file_path_jpg
#     else:
#         print(f"Image {i} not found in either png or jpg format.")
#         continue

#     # Open the image
#     image = Image.open(input_file_path)
#     print("Input Size: ", image.size)

#     # Generate variations of the image
#     variations = conditioning_model.generate_images(input_image=image, sample_count=5, steps=50, seed=42)
#     print(variations)
#     resized_variations = []

#     for k in range(len(variations)):
#       resized_image = resize_without_padding(variations[k], target_size=image.size)
#       resized_img = Image.fromarray(resized_image)
#       resized_variations.append(resized_img)
#       print("Size of Output before Resizing: ",variations[k].size)
#       print("Size of Output after Resizing: ",resized_img.size)



#     # Save the variations to the output folder
#     for j, variation in enumerate(resized_variations):
#         output_file_path = os.path.join(output_folder, f"output_{i}{chr(ord('a') + j)}.png")
#         variation.save(output_file_path)
#         print(f"Saved: {output_file_path}")

# print("Processing complete.")

# -----------------------------------------------------------------------------------------------


# Testing for a single image ---------------------------------------------------------------------

input_image_path = "/content/test/30.png"
input_image = Image.open(input_image_path)
print("Size of Input Image: ", input_image.size)

images = conditioning_model.generate_images(input_image=input_image, sample_count=5, steps=50, seed=42)
i = 0
for i in range(len(images)):
    print("Size of Output before Resizing: ",images[i].size)
    resized_image = resize_without_padding(images[i], target_size=input_image.size)
    resized_img = Image.fromarray(resized_image)
    print("Size of Output after Resizing: ",resized_img.size)
    resized_img.save(f"test_output_{i}.png")
    plt.imshow(resized_image)
    plt.axis("off")
    plt.show()

# -----------------------------------------------------------------------------------------------