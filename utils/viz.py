import cv2, torch
from PIL import Image
import numpy as np
from torchvision import transforms as vt

def draw_overlaid(original_image: Image.Image, heatmap_image: Image.Image) -> Image:
    heatmap_array = cv2.applyColorMap(np.array(heatmap_image), cv2.COLORMAP_JET)
    heatmap_array = cv2.cvtColor(heatmap_array, cv2.COLOR_BGR2RGB)
    overlaid_array = cv2.addWeighted(np.array(original_image), 0.5, heatmap_array, 0.5, 0)
    return Image.fromarray(overlaid_array)

def draw_heatmap(heatmap_image: np.array, resolution: tuple[int]) -> Image:
    heatmap_image = Image.fromarray(heatmap_image, 'L')
    heatmap_image = heatmap_image.resize(resolution, Image.BICUBIC)
    return heatmap_image
