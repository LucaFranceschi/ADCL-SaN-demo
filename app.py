import os
import torch
import torchaudio
import cv2
import subprocess
import uuid
import shutil
import base64

import numpy as np
import gradio as gr

from io import BytesIO
from typing import cast, TypedDict
from PIL import Image
from PIL.Image import Image as PImage
from importlib import import_module
from torchvision import transforms as vt

# from modules.models import ACL, ADCL
from utils.util import get_prompt_template
from utils.viz import draw_overlaid, draw_overlaid_im, draw_heatmap

# ========================================== ENV SETTINGS ==========================================

WEIGHTS_PATH = 'data/models/{}'
CONFIGS_PATH = 'config/model/{}'
PT_MODELS_PATH = 'data/pretrain'
EXAMPLES_PATH = 'data/examples'

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda', torch.cuda.current_device()) if USE_CUDA else torch.device('cpu')
print(f'Device: {DEVICE} is used\n')

# =========================================== CONSTANTS ===========================================

INPUT_RESOLUTION = 352
SAMPLE_RATE = 16000
PROMPT_TEMPLATE, TEXT_POS_AT_PROMPT, PROMPT_LENGTH = get_prompt_template()
MEDIA_DIR = 'media'

# ========================================= MODEL WRAPPER =========================================

if "MODELS" not in globals():
    global MODELS
    MODELS = {}

def cleanup():
    global MODELS
    MODELS = {}

def multiton(cls):
    global MODELS
    def getinstance(name, *args):
        if name not in MODELS:
            MODELS[name] = cls(name, *args)
        return MODELS[name]
    return getinstance

@multiton
class Model:
    def __init__(
        self,
        model_version,
        display_name: str|None = None,
        model_classname: str|None = None,
        weights_path: str|None = None,
        config_file_path: str|None = None,
        univ_threshold: float|None = None,
    ) -> None:
        self.model_version = model_version
        self.display_name = display_name
        self.model_classname = model_classname
        self.weights_path = WEIGHTS_PATH.format(weights_path)
        self.config_file_path = CONFIGS_PATH.format(config_file_path)
        self.univ_threshold = univ_threshold

        self.model = None
        self.silence_emb = None
        self.noise_emb = None

    def load_model(self):
        if self.model == None and self.model_classname and self.config_file_path:
            print('Loaded model', self.model_version)
            self.model = getattr(import_module('modules.models'), self.model_classname)(
                self.config_file_path,
                DEVICE,
                PT_MODELS_PATH
            )

            self.model.load(self.weights_path)
            self.model.train(False)

    def offload_model(self):
        if self.model != None:
            del self.model
            self.model = None

    def get_silence_emb(self) -> torch.Tensor:
        if self.silence_emb == None:
            assert(self.model != None)
            placeholder_tokens = self.model.get_placeholder_token(PROMPT_TEMPLATE.replace('{}', ''))

            self.silence_emb = self.model.encode_audio(
                torch.zeros((1, 3*SAMPLE_RATE)).to(self.model.device),
                placeholder_tokens,
                TEXT_POS_AT_PROMPT,
                PROMPT_LENGTH
            )
        return self.silence_emb

    def get_noise_emb(self) -> torch.Tensor:
        if self.noise_emb == None:
            assert(self.model != None)
            placeholder_tokens = self.model.get_placeholder_token(PROMPT_TEMPLATE.replace('{}', ''))

            self.noise_emb = self.model.encode_audio(
                torch.clip(torch.randn((1, 3*SAMPLE_RATE)), min=-1., max=1.).to(self.model.device),
                placeholder_tokens,
                TEXT_POS_AT_PROMPT,
                PROMPT_LENGTH
            )
        return self.noise_emb

    def embed_audio(self, audio) -> torch.Tensor:
        assert(self.model != None)
        placeholder_tokens = self.model.get_placeholder_token(PROMPT_TEMPLATE.replace('{}', ''))

        return self.model.encode_audio(
            audio.to(self.model.device),
            placeholder_tokens,
            TEXT_POS_AT_PROMPT,
            PROMPT_LENGTH
        )

Model('baseline', 'ACL-SSL Baseline', 'ACL', 'ACL_ViT16_test_best_param/Param_best.pth', 'ACL_ViT16.yaml', 0.870)
Model('ACL-SaN_v1_B16', 'ACL-SaN v1', 'ACL', 'ACL-SaN_v1_B16_E17.pth', 'ACL_ViT16.yaml', 0.920)
Model('ACL-SaN_v1_B32', 'ACL-SaN v1 (B32)', 'ACL', 'ACL-SaN_v1_B32_E19.pth', 'ACL_ViT16.yaml', 0.930)
Model('ACL-SaN_v2_B16', 'ACL-SaN v2', 'ACL', 'ACL-SaN_v2_B16_E16.pth', 'ACL_ViT16.yaml', 0.883)
Model('ACL-SaN_v3_B16', 'ACL-SaN v3', 'ACL', 'ACL-SaN_v3_B16_E15.pth', 'ACL_ViT16.yaml', 0.876)
Model('ACL-SaN_v4_B16', 'ACL-SaN v4', 'ACL', 'ACL-SaN_v4_B16_E18.pth', 'ACL_ViT16.yaml', 0.875)
Model('ACL-SaN_v5_B16', 'ACL-SaN v5', 'ACL', 'ACL-SaN_v5_B16_E16.pth', 'ACL_ViT16.yaml', 0.613)
Model('ADCL_vA_B16', 'ADCL vA', 'ADCL', 'ACL-SaN_v1_B16_E17.pth', 'ADCL_ViT16.yaml', 0.642)
Model('ADCL_vB_B16', 'ADCL vB', 'ADCL', 'ADCL_vB_B16_E18.pth', 'ADCL_ViT16.yaml', 0.384)
Model('ADCL_vC_B16', 'ADCL vC', 'ADCL', 'ADCL_vC_B16_E17.pth', 'ADCL_ViT16-v2.yaml', 0.842)

# ======================================= SESSION MANAGEMENT ======================================

# required fields. PEP 655 unavailable
class _SessionState(TypedDict):
    session_id: str
    session_dir: str
    comparison_segs: list[np.ndarray]
    comparison_models: list[str]

class SessionState(_SessionState, total=False):
    """Per-session state dictionary"""
    image_seg: np.ndarray
    image_resolution: tuple[int, int]
    video_seg: np.ndarray
    video_resolution: tuple[int, int]
    video_audio_path: str
    video_fps: int
    comparison_resolution: tuple[int, int]


def create_session() -> SessionState:
    """Create a new session with its own directory"""
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(MEDIA_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    print(f'Created session: {session_id} at {session_dir}')

    return SessionState(
        session_id=session_id,
        session_dir=session_dir,
        comparison_segs=[],
        comparison_models=[]
    )


def cleanup_session(state: SessionState) -> None:
    """Clean up session directory and files"""
    if 'session_dir' not in state:
        return

    session_dir = state['session_dir']
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
        print(f'Cleaned up session directory: {session_dir}')


# =========================================== FUNCTIONS ===========================================

def apply_threshold_to_segmentation(seg: np.ndarray, threshold: float, model='') -> np.ndarray:
    """Apply threshold to segmentation map"""
    seg_thresholded = np.where(seg >= threshold*255, 255, 0).astype(np.uint8)
    return seg_thresholded


# @gr.cache
def update_threshold(thr: float, state: SessionState) -> PImage:
    """Update threshold for image segmentation"""
    if 'image_seg' not in state or 'image_resolution' not in state:
        return gr.skip() # type: ignore

    seg_thresholded = apply_threshold_to_segmentation(state['image_seg'], thr)
    heatmap_mask = draw_heatmap(seg_thresholded, state['image_resolution'])
    return Image.fromarray(heatmap_mask)


# @gr.cache
def update_threshold_video(thr: float, state: SessionState) -> str:
    """Update threshold for video segmentation"""
    if 'video_seg' not in state or \
        'video_resolution' not in state or \
        'video_audio_path' not in state or \
        'video_fps' not in state:
        return gr.skip() # type: ignore

    seg_thresholded = apply_threshold_to_segmentation(state['video_seg'], thr)

    v_heatmap_mask = []
    for i in range(seg_thresholded.shape[0]):
        v_seg = seg_thresholded[i]
        v_heatmap_mask.append(draw_heatmap(v_seg, state['video_resolution']))

    heatmap_mask = save_video(
        v_heatmap_mask,
        state['video_audio_path'],
        os.path.join(state['session_dir'], 'video_mask.mp4'),
        state['video_resolution'],
        state['video_fps']
    )

    return heatmap_mask


@torch.no_grad()
def forward(
    image: torch.Tensor,
    audio: torch.Tensor | str,
    model_version: str,
) -> np.ndarray:
    """Perform a forward pass and return the raw segmentation map as numpy array (0-255)"""
    model = Model(model_version)
    model.load_model() # imports and loads weights if not done before
    module = model.model # get handle of module instance
    assert(module != None)

    if audio == 'silence':
        audio_driven_embedding = model.get_silence_emb()
    elif audio == 'noise':
        audio_driven_embedding = model.get_noise_emb()
    else:
        audio_driven_embedding = model.embed_audio(audio)

    out_dict = module(image.to(DEVICE), resolution=INPUT_RESOLUTION, pred_emb=audio_driven_embedding)

    if model_version == 'ADCL_vA_B16':
        seg = ((out_dict['positive']['v_d_seg'].squeeze().cpu().numpy()) * 255).astype(np.uint8)
    else:
        seg = ((out_dict['positive']['m_i_seg'].squeeze().cpu().numpy()) * 255).astype(np.uint8)

    return seg


def submit(
    image_file: PImage,
    audio_file: tuple[int, np.ndarray] | str,
    model_name: str,
    model_version: str,
    threshold: float,
    state: SessionState,
    comparison_flag: bool = False
) -> tuple[PImage, PImage, SessionState]:
    """Submit image + audio and return heatmap and overlaid visualization"""
    original_resolution = image_file.size
    resolution = min(original_resolution)

    image_transform = vt.Compose([
        vt.Resize((resolution, resolution), vt.InterpolationMode.BICUBIC),
        vt.ToTensor(),
        vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
    ])
    image = cast(torch.Tensor, image_transform(image_file))

    # simulate batch dimension
    image = image.unsqueeze(0)

    if type(audio_file) == str:
        audio = audio_file
        assert(type(audio) == str)
    else:
        sr, audio = audio_file
        assert(type(audio) == np.ndarray)
        needs_normalization = audio.dtype == np.int16
        audio = torch.Tensor(audio)

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        if audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            audio = audio.T

        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(cast(int, sr), SAMPLE_RATE)
            audio = resampler(audio)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        # audios can be normalized to -1,1 OR occupy the whole np.int16 range depending on the loading method
        if needs_normalization:
            audio = audio / np.iinfo(np.int16).max
        assert(type(audio) == torch.Tensor)

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


@torch.no_grad()
def forward_video(
    frames: torch.Tensor,
    audio: torch.Tensor,
    model_version: str,
) -> np.ndarray:
    """Perform forward pass on video frames"""
    model = Model(model_version)
    model.load_model() # imports and loads weights if not done before
    module = model.model # get handle of module instance
    assert(module != None)

    placeholder_tokens = module.get_placeholder_token(PROMPT_TEMPLATE.replace('{}', ''))

    audio_driven_embedding = module.encode_audio(
        audio.to(module.device),
        placeholder_tokens,
        TEXT_POS_AT_PROMPT,
        PROMPT_LENGTH
    )

    v_seg = []
    for i in range(frames.shape[0]):
        out_dict = module(
            frames[i].unsqueeze(0).to(DEVICE),
            resolution=INPUT_RESOLUTION,
            pred_emb=audio_driven_embedding
        )

        if model_version == 'ADCL_vA_B16':
            seg = ((out_dict['positive']['v_d_seg'].squeeze().cpu().numpy()) * 255).astype(np.uint8)
        else:
            seg = ((out_dict['positive']['m_i_seg'].squeeze().cpu().numpy()) * 255).astype(np.uint8)

        v_seg.append(seg)

    return np.array(v_seg)


def save_video(
    video_frames: list[np.ndarray],
    audio_path: str,
    output_path: str,
    original_resolution: tuple[int, int],
    fps: int
) -> str:
    """Save video frames with audio to file"""
    # Write to temp AVI with OpenCV
    temp_video = os.path.join(os.path.dirname(output_path), str(uuid.uuid4()) + '.avi')

    video = cv2.VideoWriter(
        temp_video,
        cv2.VideoWriter.fourcc(*'MJPG'),
        fps,
        original_resolution
    )

    for i in range(len(video_frames)):
        frame = video_frames[i]

        if len(frame.shape) == 2:  # Grayscale (H, W)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:  # RGB image (H, W, 3)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:  # Already BGR or other format
            frame_bgr = frame

        video.write(frame_bgr)

    video.release()

    # Convert to browser-compatible MP4 using ffmpeg
    subprocess.run(
        [
            'ffmpeg', '-y', '-i', temp_video, '-i', audio_path, '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '23', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_path
        ],
        capture_output=True,
        check=True
    )

    os.remove(temp_video)

    return output_path


def submit_video(
    video_file: str,
    model_name: str,
    model_version: str,
    threshold: float,
    state: SessionState
) -> tuple[str, str, SessionState]:
    """Submit video and return heatmap and overlaid visualization"""
    # Extract video frames
    video = cv2.VideoCapture(video_file)
    original_resolution = (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    fps = int(video.get(cv2.CAP_PROP_FPS))

    resolution = min(original_resolution)
    image_transform = vt.Compose([
        vt.Resize((resolution, resolution), vt.InterpolationMode.BICUBIC),
        vt.ToTensor(),
        vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
    ])

    original_frames = []
    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        original_frames.append(frame)
        frames.append(cast(torch.Tensor, image_transform(frame)))
    video.release()

    frames = torch.stack(frames)

    # Extract audio using ffmpeg subprocess
    audio_path = os.path.join(state['session_dir'], 'extracted_audio.wav')

    try:
        subprocess.run(
            [
                'ffmpeg', '-i', video_file, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000',
                '-ac', '1', '-y', audio_path
            ],
            capture_output=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise

    # Load extracted audio
    audio, sr = torchaudio.load(audio_path)  # type: ignore

    # Resample if needed
    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio = resampler(audio)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)

    video_file_name = video_file.split('/')[-1]

    if any(video_file_name.startswith(cat) for cat in ['original', 'silence', 'noise', 'offscreen']):
        likely_output_path = os.path.join(
            'data/examples/v_seg',
            video_file_name.removesuffix('.mp4') + '_' + '_'.join([model_version, 'v_seg.npy'])
        )
        if os.path.exists(likely_output_path):
            v_seg = np.load(likely_output_path)
        else:
            v_seg = forward_video(frames, audio, model_version)
            np.save(likely_output_path, v_seg)
    else:
        v_seg = forward_video(frames, audio, model_version)

    # Store in state
    state['video_seg'] = v_seg
    state['video_resolution'] = original_resolution
    state['video_audio_path'] = audio_path
    state['video_fps'] = fps

    model = Model(model_version)
    assert(model.univ_threshold != None)

    # Create overlaid image
    v_overlaid = []
    v_heatmap_mask = []
    for i in range(v_seg.shape[0]):
        seg = v_seg[i]
        heatmap = draw_heatmap(seg, original_resolution)
        v_overlaid.append(draw_overlaid(np.array(original_frames[i]), heatmap))

        # Apply threshold
        seg_thresholded = apply_threshold_to_segmentation(seg, model.univ_threshold)
        v_heatmap_mask.append(draw_heatmap(seg_thresholded, original_resolution))

    overlaid = save_video(
        v_overlaid,
        audio_path,
        os.path.join(state['session_dir'], 'video_overlaid.mp4'),
        original_resolution,
        fps
    )

    heatmap_mask = save_video(
        v_heatmap_mask,
        audio_path,
        os.path.join(state['session_dir'], 'video_mask.mp4'),
        original_resolution,
        fps
    )

    return heatmap_mask, overlaid, state


# ==================================== EXAMPLE VIDEO MANAGEMENT ===================================

def load_example_videos() -> dict[str, list[str]]:
    """
    Load example videos from data/examples directories.
    Returns a dictionary organized by category.

    Returns:
        dict: {
            'Original': [list of paths],
            'Silence': [list of paths],
            'Noise': [list of paths],
            'Offscreen (Swapped Audio)': [list of paths]
        }
    """
    examples_map = ['original', 'silence', 'noise', 'offscreen']

    examples_dict = {}

    for category in examples_map:
        examples_dict[category] = []

        if os.path.exists(EXAMPLES_PATH):
            video_files = sorted([
                os.path.join(EXAMPLES_PATH, f) for f in os.listdir(EXAMPLES_PATH) if f.startswith(category)
            ])
            examples_dict[category] = video_files

    return examples_dict


def organize_examples_for_gradio(examples_dict: dict[str, list[str]]) -> list[list[str]]:
    """
    Organize examples for Gradio in tabular format.
    Each row represents one video across all categories.

    Args:
        examples_dict: Dictionary of {category: [video_paths]}

    Returns:
        List of examples in format [[video_path, category_label], ...]
    """
    examples_map = ['original', 'silence', 'noise', 'offscreen']
    examples_list = []

    for category in examples_map:
        examples_list += examples_dict[category]

    return examples_list

def update_comparison_type(image_file: PImage, output_type: str, state: SessionState) -> str:
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
        return _render_comparison_html(grid, col_names)
    else:
        return update_comparison_threshold(output_type, state)

def update_comparison_threshold(output_type: str, state: SessionState) -> str:
    """Update threshold for image segmentation"""
    if 'comparison_segs' not in state or 'comparison_resolution' not in state or 'comparison_models' not in state or output_type == 'Overlaid':
        return gr.skip() # type: ignore

    grid = []
    for i in range(len(state['comparison_segs'])):
        seg_thresholded = apply_threshold_to_segmentation(state['comparison_segs'][i], Model(state['comparison_models'][i//3]).univ_threshold) #type:ignore
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
    state: SessionState
) -> tuple[str, SessionState]:

    state['comparison_segs'] = []
    state['comparison_models'] = []

    overlaid_list = []
    seg_masks_list = []
    col_labels = []
    comparison_models = []
    thresholds = []

    original_resolution = image_file.size

    for display_name, model_version in CHOICES_VERSIONS[model_name]:
        model = Model(model_version)
        model.load_model()
        assert(model.univ_threshold != None)

        col_labels.append(display_name)
        comparison_models.append(model_version)
        thresholds.append(model.univ_threshold)

        for audio in [audio_file, "silence", "noise"]:
            mask, overlaid, state = submit(
                image_file, audio, model_name, model_version,
                model.univ_threshold, state, True
            )
            overlaid_list.append(overlaid)
            seg_masks_list.append(mask)

        model.offload_model()

    state['comparison_resolution'] = original_resolution
    state['comparison_models'] = comparison_models

    grid = overlaid_list if output_type == "Overlaid" else seg_masks_list
    return _render_comparison_html(grid, col_labels), state

def _render_comparison_html(grid: list[PImage], col_labels: list[str]) -> str:
    row_labels = ["Audio", "Silence", "Noise"]
    images = [pil_to_base64(im) for im in grid]
    return images_to_html(images, col_labels=col_labels, row_labels=row_labels)

# ========================================== APPLICATION ==========================================

title = "Audio-Grounded Contrastive Learning"

CHOICES_MODELS = ['ACL-SaN', 'ADCL']

CHOICES_VERSIONS = {
    'ACL-SaN': [
        ('ACL-SSL Baseline', 'baseline'),
        ('ACL-SaN v1', 'ACL-SaN_v1_B16'),
        ('ACL-SaN v1 (B32)', 'ACL-SaN_v1_B32'),
        ('ACL-SaN v2', 'ACL-SaN_v2_B16'),
        ('ACL-SaN v3', 'ACL-SaN_v3_B16'),
        ('ACL-SaN v4', 'ACL-SaN_v4_B16'),
        ('ACL-SaN v5', 'ACL-SaN_v5_B16')
    ],
    'ADCL': [
        ('ACL-SSL Baseline', 'baseline'),
        ('ACL-SaN v1', 'ACL-SaN_v1_B16'),
        ('ADCL vA', 'ADCL_vA_B16'),
        ('ADCL vB', 'ADCL_vB_B16'),
        ('ADCL vC', 'ADCL_vC_B16'),
    ]
}
choices_models_init = CHOICES_MODELS[0]

EMPTY_COMPARISON_TABLE = """
<div style="display:none">
<table>
<tr><td>placeholder</td></tr>
</table>
</div>
"""

def update_versions(model_name):
    return gr.Dropdown(
        choices=CHOICES_VERSIONS[model_name],
        value=CHOICES_VERSIONS[model_name][0][1]
    )

def pil_to_base64(img):
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

def images_to_html(images, col_labels, row_labels):
    html = f"""
    <style>
    .comparison-table,
    .comparison-table *,
    .comparison-table tr,
    .comparison-table td,
    .comparison-table th,
    .comparison-table tbody,
    .comparison-table thead {{
        border: 0 !important;
        outline: none !important;
        box-shadow: none !important;
    }}

    .comparison-table {{
        width: 100% !important;
        table-layout: fixed !important;
        border-collapse: collapse !important;
        border-spacing: 0 !important;
        font-family: var(--font, ui-sans-serif, system-ui, sans-serif) !important;
        color: var(--body-text-color) !important;
    }}

    /* HARD FORCE COLUMN WIDTHS */
    .comparison-table col.label-col {{
        width: 24px !important;
    }}

    .comparison-table th,
    .comparison-table td {{
        padding: 2px !important;
        text-align: center !important;
        vertical-align: middle !important;
        border: none !important;
        line-height: 0 !important;
    }}
    .comparison-table th {{
        font-weight: 600 !important;
        font-size: 1rem !important;
        line-height: normal !important;
        padding-bottom: 6px !important;
    }}

    /* tiny first column */
    .comparison-table td.row-label,
    .comparison-table th.row-label {{
        width: 24px !important;
        min-width: 24px !important;
        max-width: 24px !important;
        padding: 0 !important;
        overflow: visible !important;
    }}

    .row-label-inner {{
        writing-mode: vertical-rl;
        transform: rotate(180deg);

        font-weight: 600;
        white-space: nowrap;

        width: 24px;
        margin: 0 auto;
    }}

    .img-container {{
        width: 100% !important;
        display: block !important;
        overflow: hidden !important;
        border-radius: 12px !important;
        font-size: 0 !important;
    }}
    .img-container img {{
        width: 100% !important;
        height: auto !important;
        max-width: 100% !important;
        object-fit: fill !important;
        display: block !important;
        aspect-ratio: 1 / 1 !important;
    }}
    </style>

    <table class="comparison-table">

        <colgroup>
            <col class="label-col">
            {"".join("<col>" for _ in range(len(col_labels)))}
        </colgroup>

        <thead>
            <tr>
                <th class="row-label"></th>
    """

    for label in col_labels:
        html += f"<th>{label}</th>"

    html += "</tr></thead><tbody>"

    for row_idx, row_label in enumerate(row_labels):
        html += f"""
        <tr>
            <td class="row-label">
                <div class="row-label-inner">{row_label}</div>
            </td>
        """

        for col_idx in range(len(col_labels)):
            img_src = images[col_idx * len(row_labels) + row_idx]

            html += f"""
            <td>
                <div class="img-container">
                    <img src="{img_src}">
                </div>
            </td>
            """

        html += "</tr>"

    html += "</tbody></table>"

    return html

with gr.Blocks() as demo:
    # Initialize session state per client
    session_state = gr.State(delete_callback=cleanup_session)
    demo.load(fn=create_session, outputs=session_state)

    with gr.Tabs():
        with gr.TabItem("Model comparisons"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        model_name_in_comp = gr.Dropdown(choices=CHOICES_MODELS, value=choices_models_init, label="Model")
                        output_type_toggle = gr.Dropdown(
                            choices=["Overlaid", "Segmentation Mask"],
                            value="Overlaid",
                            label="Output type"
                        )
                    image_in_comp = gr.Image(type='pil', label="Image Input", height=350)
                    audio_in_comp = gr.Audio(label="Audio Input")
                    btn_comp = gr.Button("Run")

                with gr.Column(scale=4):
                    comp_html_out = gr.HTML(
                        value='<div style="height:20% !important;opacity:0 !important"></div>'
                    )

            btn_comp.click(
                fn=submit_comparison,
                inputs=[image_in_comp, audio_in_comp, model_name_in_comp,
                        output_type_toggle, session_state],
                outputs=[comp_html_out, session_state]
            )
            output_type_toggle.change(
                fn=update_comparison_type,
                inputs=[image_in_comp, output_type_toggle, session_state],
                outputs=comp_html_out
            )

        # ============= IMAGE + AUDIO TAB =============
        with gr.TabItem("Image + Audio"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_name_in = gr.Dropdown(choices=CHOICES_MODELS, value=choices_models_init, label="Model")
                        model_version_name_in = gr.Dropdown(
                            choices=CHOICES_VERSIONS[choices_models_init],
                            value=CHOICES_VERSIONS[choices_models_init][0][1],  # 'baseline'
                            label="Version"
                        )
                        model_name_in.change(fn=update_versions, inputs=model_name_in, outputs=model_version_name_in)

                    image_in = gr.Image(type='pil', label="Image Input", height=350)
                    audio_in = gr.Audio(label="Audio Input")
                    btn = gr.Button("Run")

                with gr.Column():
                    overlaid_out = gr.Image(type='pil', label="Overlaid with Original", height=350)
                    heatmap_out = gr.Image(type='pil', label="Heatmap (Grayscale)", height=350)
                    threshold_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.01,
                        label="Threshold",
                        info="Default value is universal threshold",
                        interactive=False
                    )

            btn.click(
                fn=submit,
                inputs=[image_in, audio_in, model_name_in, model_version_name_in, threshold_slider, session_state],
                outputs=[heatmap_out, overlaid_out, session_state]
            ).then(
                fn=lambda model_version: gr.update(value=Model(model_version).univ_threshold, interactive=True),  # Enable slider after results
                inputs=[model_version_name_in],
                outputs=threshold_slider
            )

            threshold_slider.change(
                fn=update_threshold,
                inputs=[threshold_slider, session_state],
                outputs=heatmap_out
            )

        # ============= VIDEO TAB =============
        with gr.TabItem("Video"):
            # Load example videos
            example_videos_dict = load_example_videos()
            example_videos_count = sum(len(v) for v in example_videos_dict.values())

            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        model_name_in_video = gr.Dropdown(choices=CHOICES_MODELS, value=choices_models_init, label="Model")
                        model_version_name_in_video = gr.Dropdown(
                            choices=CHOICES_VERSIONS[choices_models_init],
                            value=CHOICES_VERSIONS[choices_models_init][0][1],  # 'baseline'
                            label="Version"
                        )
                        model_name_in_video.change(fn=update_versions, inputs=model_name_in_video, outputs=model_version_name_in_video)

                    video_in = gr.Video(label="Video Input", height=350)
                    # Display examples if available
                    if example_videos_count > 0:
                        gr.Markdown(f"**Available examples:** {' | '.join([f'{cat} ({len(vids)})' for cat, vids in example_videos_dict.items() if vids])}")
                        example_videos = organize_examples_for_gradio(example_videos_dict)

                        if example_videos:
                            gr.Examples(
                                examples=example_videos,
                                inputs=[video_in],
                                label="Click to load an example video",
                                examples_per_page=5
                            )
                    btn_video = gr.Button("Run")
                with gr.Column():
                    v_overlaid_out = gr.Video(label="Overlaid with Original", height=350)
                    v_heatmap_out = gr.Video(label="Heatmap (Grayscale)", height=350)
                    threshold_slider_video = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=0.5,
                        step=0.01,
                        label="Threshold",
                        info="Default value is universal threshold",
                        interactive=False
                    )

            btn_video.click(
                fn=submit_video,
                inputs=[video_in, model_name_in_video, model_version_name_in_video, threshold_slider_video, session_state],
                outputs=[v_heatmap_out, v_overlaid_out, session_state]
            ).then(
                fn=lambda model_version: gr.update(value=Model(model_version).univ_threshold, interactive=True),
                inputs=[model_version_name_in_video],
                outputs=threshold_slider_video
            )

            threshold_slider_video.change(
                fn=update_threshold_video,
                inputs=[threshold_slider_video, session_state],
                outputs=v_heatmap_out
            )

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
