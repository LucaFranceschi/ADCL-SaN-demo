import base64

import numpy as np

from PIL import Image
from PIL.Image import Image as PImage
from typing import cast
from io import BytesIO
import gradio as gr

from .classic import submit
from ..model import Model, MODEL_REGISTRY
from ..constants import *
from ..session import SessionState
from ..front_utils import images_to_html

from utils.viz import draw_overlaid_im, draw_heatmap

# ==================================== COMPARISON TAB FUNCTIONS ====================================

def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

def update_comparison_type(
    image_file: PImage,
    output_type: str,
    thresh_type: str,
    thresh_value: float,
    state: SessionState
) -> tuple[str, dict, dict]:
    if 'comparison_segs' not in state or 'comparison_resolution' not in state or 'comparison_models' not in state:
        return gr.skip() # type: ignore

    if output_type == 'Overlaid':
        grid = []
        for i in range(len(state['comparison_segs'])):
            heatmap = draw_heatmap(state['comparison_segs'][i], state['comparison_resolution'])
            overlaid = draw_overlaid_im(image_file, Image.fromarray(heatmap))
            grid.append(overlaid)

        col_names = [Model(model).display_name for model in state['comparison_models']]
        col_names = cast(list[str], col_names)
        return _render_comparison_html(grid, col_names), gr.update(interactive=False), gr.update(interactive=False)
    else:
        return update_comparison_threshold(output_type, thresh_type, thresh_value, state), gr.update(interactive=True), gr.update(interactive=True)

def update_comparison_threshold(
    output_type: str,
    thresh_type: str,
    thresh_value: float,
    state: SessionState
) -> str:
    """Update threshold for image segmentation"""
    if 'comparison_segs' not in state or 'comparison_resolution' not in state or 'comparison_models' not in state or output_type == 'Overlaid':
        return gr.skip() # type: ignore

    grid = []
    for i in range(len(state['comparison_segs'])):
        used_thresh = None
        if thresh_type == 'custom':
            used_thresh = thresh_value
        else:
            used_thresh = eval(f'Model(state["comparison_models"][i//3]).{thresh_type}')
            assert(used_thresh != None)

        seg_thresholded = apply_threshold_to_segmentation(state['comparison_segs'][i], used_thresh) #type:ignore
        heatmap_mask = draw_heatmap(seg_thresholded, state['comparison_resolution'])
        grid.append(Image.fromarray(heatmap_mask))

    col_names = [Model(model).display_name for model in state['comparison_models']]
    col_names = cast(list[str], col_names)
    return _render_comparison_html(grid, col_names)


def submit_comparison(
    image_file: PImage,
    audio_file: tuple[int, np.ndarray],
    model_name: str,
    output_type: str,
    thresh_type: str,
    thresh_value: float,
    state: SessionState,
    progress=gr.Progress()
) -> tuple[str, SessionState]:

    state['comparison_segs'] = []
    state['comparison_models'] = []

    overlaid_list = []
    seg_masks_list = []
    col_labels = []
    comparison_models = []

    original_resolution = image_file.size

    model_version_choices = CHOICES_VERSIONS[model_name]

    total_steps = len(model_version_choices)*3
    step = 0

    for display_name, model_version in model_version_choices:
        model = Model(model_version)
        model.load_model()
        used_thresh = None
        if thresh_type == 'custom':
            used_thresh = thresh_value
        else:
            used_thresh = eval(f'model.{thresh_type}')
            assert(used_thresh != None)

        col_labels.append(display_name)
        comparison_models.append(model_version)

        for audio in [audio_file, "silence", "noise"]:
            progress(step / total_steps, desc=f"Running {display_name}...")
            mask, overlaid, state = submit(
                image_file, audio, model_name, model_version,
                used_thresh, state, True
            )
            overlaid_list.append(overlaid)
            seg_masks_list.append(mask)
            step += 1

        model.offload_model() # TODO: REMOVE BEFORE FINAL RELEASE

    progress(1.0, desc="Done!")
    state['comparison_resolution'] = original_resolution
    state['comparison_models'] = comparison_models

    grid = overlaid_list if output_type == "Overlaid" else seg_masks_list
    return _render_comparison_html(grid, col_labels), state

def _render_comparison_html(grid: list[PImage], col_labels: list[str]) -> str:
    row_labels = ["Audio", "Silence", "Noise"]
    images = [pil_to_base64(im) for im in grid]
    return images_to_html(images, col_labels=col_labels, row_labels=row_labels)

# Build CHOICES_VERSIONS from the registry
CHOICES_VERSIONS = {}
for key, cfg in MODEL_REGISTRY.items():
    CHOICES_VERSIONS.setdefault(cfg['group'], []).append((cfg['display_name'], key))