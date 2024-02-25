from preprocess import preprocess_image
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import os
from PIL import Image

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
CACHE_DIR = os.path.join(ROOT_DIR, ".cache")

os.makedirs(CACHE_DIR, exist_ok=True)
pipe_id = "runwayml/stable-diffusion-inpainting"

canny_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny",cache_dir=CACHE_DIR)
depth_controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth",cache_dir=CACHE_DIR)


pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
    pipe_id, controlnet=[canny_controlnet, depth_controlnet], cache_dir=CACHE_DIR
)


image_path = "inputs/image1.jpeg"
obj_img, mask_img, canny_img, depth_img = preprocess_image(image_path)


pipe.load_lora_weights("animemix_v3_offset.safetensors", adapter_name="anime")

pipe.set_adapters(["anime"], adapter_weights=[1.0])


prompt = ""
image = pipe(
    prompt,
    num_inference_steps=10,
    cross_attention_kwargs={"scale": 1.0},
    generator=torch.manual_seed(0),
    image=Image.open(image_path),
    mask_image=mask_img,
    control_image = [canny_img, depth_img]
).images[0]

image.save("output.png")
