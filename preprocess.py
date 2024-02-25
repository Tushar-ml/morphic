from rembg import remove
from rembg.session_factory import new_session
from PIL import Image
import numpy as np
from copy import deepcopy
import os
import cv2
import torch, matplotlib
from typing import Tuple

image_path = "inputs/image1.jpeg"
repo = "isl-org/ZoeDepth"
model_zoe_nk = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
session = new_session("isnet-general-use")

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):

    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def preprocess_image(image_path: str):
    filename, ext = os.path.splitext(os.path.basename(image_path))

    os.makedirs(f"outputs/{filename}", exist_ok=True)

    image = np.asarray(Image.open(image_path))
    mask_image = remove(image, only_mask=True, session=session)

    bg_img = deepcopy(image)
    bg_img[mask_image != 0] = 0

    obj_img = deepcopy(image)
    obj_img[mask_image == 0] = 0

    bg_img = numpy_to_pil(bg_img)
    obj_img = numpy_to_pil(obj_img)
    mask_image = numpy_to_pil(mask_image)

    cannyImg = cv2.Canny(np.asarray(obj_img),100,200)
    canny_img = numpy_to_pil(cannyImg)

    depth_img = find_depth(obj_img)

    return obj_img, mask_image, canny_img, depth_img

def numpy_to_pil(image: np.ndarray) -> Image.Image:
    return Image.fromarray(image)

def find_depth(img: Image.Image) -> Image.Image:
    depth_numpy = model_zoe_nk.infer_pil(img)
    depth_numpy = colorize(depth_numpy)

    return numpy_to_pil(depth_numpy).convert("RGB")

preprocess_image(image_path)