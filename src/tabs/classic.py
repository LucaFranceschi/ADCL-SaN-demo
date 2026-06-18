import os

import torch
import torchaudio
import numpy as np

from PIL import Image
from PIL.Image import Image as PImage
from torchvision import transforms as vt
from typing import cast
from gradio import skip

from ..model import Model, load_audio, add_noise
from ..constants import *
from ..session import SessionState

from utils.viz import draw_overlaid_im, draw_heatmap

# ====================================== CLASSIC TAB FUNCTIONS =====================================

@torch.no_grad()
def forward(
    image: torch.Tensor,
    audio: torch.Tensor,
    model_version: str,
) -> np.ndarray:
    """Perform a forward pass and return the raw segmentation map as numpy array (0-255)"""
    model = Model(model_version)
    model.load_model() # imports and loads weights if not done before
    module = model.model # get handle of module instance
    assert(module != None)

    audio_driven_embedding = model.embed_audio(audio)

    out_dict = module(image.to(DEVICE), resolution=INPUT_RESOLUTION, pred_emb=audio_driven_embedding)

    if model_version == 'ADCL_vA_B16':
        seg = ((out_dict['positive']['v_d_seg'].squeeze().cpu().numpy()) * 255).astype(np.uint8)
    else:
        seg = ((out_dict['positive']['m_i_seg'].squeeze().cpu().numpy()) * 255).astype(np.uint8)

    return seg


def submit(
    image_file: PImage,
    audio_file: tuple[int, np.ndarray] | str | torch.Tensor,
    model_name: str,
    model_version: str,
    threshold: float,
    state: SessionState,
    comparison_flag: bool = False
) -> tuple[PImage, PImage, SessionState]:
    """Submit image + audio and return heatmap and overlaid visualization"""
    original_resolution = image_file.size

    image_transform = vt.Compose([
        vt.Resize((INPUT_RESOLUTION, INPUT_RESOLUTION), vt.InterpolationMode.BICUBIC),
        vt.ToTensor(),
        vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
    ])
    image = cast(torch.Tensor, image_transform(image_file))

    # simulate batch dimension
    image = image.unsqueeze(0)

    audio = load_audio(audio_file)

    # Get raw segmentation
    seg = forward(image, audio, model_version)

    # Store in state
    if not comparison_flag:
        state['image_seg'] = seg
        state['image_resolution'] = original_resolution
    else:
        state['comparison_segs'].append(seg)
        state['comparison_resolution'] = original_resolution

    # Create overlaid image
    heatmap = draw_heatmap(seg, original_resolution)
    overlaid = draw_overlaid_im(image_file, Image.fromarray(heatmap))

    # Apply threshold
    seg_thresholded = apply_threshold_to_segmentation(seg, threshold)
    heatmap_mask = Image.fromarray(draw_heatmap(seg_thresholded, original_resolution))

    return heatmap_mask, overlaid, state

# @gr.cache
def update_threshold(
    thresh_type: str,
    thresh_value: float,
    model_version: str,
    state: SessionState
) -> PImage:
    """Update threshold for image segmentation"""
    if 'image_seg' not in state or 'image_resolution' not in state:
        return skip() # type: ignore

    used_threshold = 0.5
    if thresh_type == 'custom':
        used_threshold = thresh_value
    elif thresh_type == 'top50p':
        used_threshold = cast(float, np.median(state['image_seg']))
    elif hasattr(Model(model_version), thresh_type):
        used_threshold = getattr(Model(model_version), thresh_type)

    seg_thresholded = apply_threshold_to_segmentation(state['image_seg'], used_threshold)
    heatmap_mask = draw_heatmap(seg_thresholded, state['image_resolution'])
    return Image.fromarray(heatmap_mask)

def load_example_frames() -> list[str]:
    if os.path.exists(FRAMES_EXAMPLES_PATH):
        return sorted([
            os.path.join(FRAMES_EXAMPLES_PATH, f) for f in os.listdir(FRAMES_EXAMPLES_PATH)
        ])
    return []

def load_example_audio() -> list[str]:
    if os.path.exists(AUDIOS_EXAMPLES_PATH):
        return [
            os.path.join(AUDIOS_EXAMPLES_PATH, f)
            for f in ['bassoon.wav', 'roar.wav', 'chew.wav', 'silence.wav', 'noise.wav']
        ]
    return []

def apply_snr(
    audio_file: tuple[int, np.ndarray] | str | torch.Tensor,
    snr: str,
    state: SessionState
) -> tuple[int, np.ndarray]:
    if not ('original_audio' in state and state["original_audio"] is not None):
        return skip() #type: ignore

    audio = load_audio(state['original_audio'])

    if snr == 'inf':
        noisy_audio = audio
    else:
        noise = torch.clip(torch.randn(audio.shape), min=-1., max=1.)
        noisy_audio = add_noise(audio, noise, torch.Tensor([float(snr)]))

    noisy_audio_np = (noisy_audio * np.iinfo(np.int16).max).numpy().astype(np.int16)[0]

    return SAMPLE_RATE, noisy_audio_np
