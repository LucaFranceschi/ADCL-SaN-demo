import cv2, torch
from PIL import Image
from PIL.Image import Image as PImage
import numpy as np
from torchvision import transforms as vt

def draw_overlaid(original_image: np.ndarray, heatmap_image: np.ndarray) -> np.ndarray:
    heatmap_array = cv2.applyColorMap(heatmap_image, cv2.COLORMAP_JET)
    heatmap_array = cv2.cvtColor(heatmap_array, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_image, 0.5, heatmap_array, 0.5, 0)

def draw_overlaid_im(original_image: PImage, heatmap_image: PImage) -> PImage:
    overlaid_array = draw_overlaid(np.array(original_image), np.array(heatmap_image))
    return Image.fromarray(overlaid_array)

def draw_heatmap(heatmap_image: np.ndarray, resolution: tuple[int, int]) -> np.ndarray:
    heatmap_result = Image.fromarray(heatmap_image, 'L')
    heatmap_result = heatmap_result.resize(resolution, Image.Resampling.BICUBIC)
    return np.array(heatmap_result)
