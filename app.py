import os
import torch
import torchaudio
import cv2
import subprocess
import uuid
import shutil

import numpy as np
import gradio as gr

from typing import cast, TypedDict
from PIL import Image
from PIL.Image import Image as PImage
from importlib import import_module
from torchvision import transforms as vt

# from modules.models import ACL, ADCL
from utils.util import get_prompt_template
from utils.viz import draw_overlaid, draw_overlaid_im, draw_heatmap


# =========================================== CONSTANTS ===========================================

INPUT_RESOLUTION = 352
SAMPLE_RATE = 16000

MODEL_PATH = 'data/pretrain'
CONFIG_FILE_TEMPLATE = 'config/model/{}_ViT16.yaml'
WEIGHTS_PATH = 'data/models'
WEIGHTS_SUBPATH = {
    'baseline': 'ACL_ViT16_test_best_param/Param_best.pth'
}

MEDIA_DIR = 'media'

USE_CUDA = torch.cuda.is_available()

PROMPT_TEMPLATE, TEXT_POS_AT_PROMPT, PROMPT_LENGTH = get_prompt_template()

DEVICE = torch.device('cuda', torch.cuda.current_device()) if USE_CUDA else torch.device('cpu')
print(f'Device: {DEVICE} is used\n')


# ======================================= SESSION MANAGEMENT ======================================

# required fields. PEP 655 unavailable
class _SessionState(TypedDict):
    session_id: str
    session_dir: str


class SessionState(_SessionState, total=False):
    """Per-session state dictionary"""
    image_seg: np.ndarray
    image_resolution: tuple[int, int]
    video_seg: np.ndarray
    video_resolution: tuple[int, int]
    video_audio_path: str
    video_fps: int


def create_session() -> SessionState:
    """Create a new session with its own directory"""
    session_id = str(uuid.uuid4())[:8]
    session_dir = os.path.join(MEDIA_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    print(f'Created session: {session_id} at {session_dir}')

    return SessionState(
        session_id=session_id,
        session_dir=session_dir
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

def apply_threshold_to_segmentation(seg: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold to segmentation map"""
    seg_thresholded = np.where(seg >= threshold*255, 255, 0).astype(np.uint8)
    return seg_thresholded


@gr.cache
def update_threshold(thr: float, state: SessionState) -> PImage:
    """Update threshold for image segmentation"""
    if 'image_seg' not in state or 'image_resolution' not in state:
        return gr.skip() # type: ignore

    seg_thresholded = apply_threshold_to_segmentation(state['image_seg'], thr)
    heatmap_mask = draw_heatmap(seg_thresholded, state['image_resolution'])
    return Image.fromarray(heatmap_mask)


@gr.cache
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


@gr.cache
@torch.no_grad()
def forward(
    image: torch.Tensor,
    audio: torch.Tensor,
    model_name: str,
    model_version: str,
    original_resolution: tuple[int, int]
) -> np.ndarray:
    """Perform a forward pass and return the raw segmentation map as numpy array (0-255)"""
    model = getattr(import_module('modules.models'), model_name)(
        CONFIG_FILE_TEMPLATE.format(model_name),
        DEVICE,
        MODEL_PATH
    )

    model.load(os.path.join(WEIGHTS_PATH, WEIGHTS_SUBPATH[model_version]))
    model.train(False)

    placeholder_tokens = model.get_placeholder_token(PROMPT_TEMPLATE.replace('{}', ''))

    resolution = min(original_resolution)

    audio_driven_embedding = model.encode_audio(
        audio.to(model.device),
        placeholder_tokens,
        TEXT_POS_AT_PROMPT,
        PROMPT_LENGTH
    )

    out_dict = model(image.to(DEVICE), resolution=INPUT_RESOLUTION, pred_emb=audio_driven_embedding)

    seg = ((out_dict['positive'].squeeze().cpu().numpy()) * 255).astype(np.uint8)

    return seg


def submit(
    image_file: PImage,
    audio_file: tuple[int, np.ndarray],
    model_name: str,
    model_version: str,
    threshold: float,
    state: SessionState
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
    print(f'{image.shape=}')

    sr, audio = audio_file
    audio = torch.Tensor(audio).T
    print(f'{audio.shape=}')

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio = resampler(audio)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)

    # simulate batch dimension
    audio = audio.unsqueeze(0)
    print(f'{audio.shape=}', f'sample_rate {sr} --> {SAMPLE_RATE}' if sr != SAMPLE_RATE else '')

    # Get raw segmentation
    seg = forward(image, audio, model_name, model_version, original_resolution)

    # Store in state
    state['image_seg'] = seg
    state['image_resolution'] = original_resolution

    # Create overlaid image
    heatmap = draw_heatmap(seg, original_resolution)
    overlaid = draw_overlaid_im(image_file, Image.fromarray(heatmap))

    # Apply threshold
    seg_thresholded = apply_threshold_to_segmentation(seg, threshold)
    heatmap_mask = Image.fromarray(draw_heatmap(seg_thresholded, original_resolution))

    return heatmap_mask, overlaid, state


@gr.cache
@torch.no_grad()
def forward_video(
    frames: torch.Tensor,
    audio: torch.Tensor,
    model_name: str,
    model_version: str,
    original_resolution: tuple[int, int]
) -> np.ndarray:
    """Perform forward pass on video frames"""
    model = getattr(import_module('modules.models'), model_name)(
        CONFIG_FILE_TEMPLATE.format(model_name),
        DEVICE,
        MODEL_PATH
    )

    model.load(os.path.join(WEIGHTS_PATH, WEIGHTS_SUBPATH[model_version]))
    model.train(False)

    placeholder_tokens = model.get_placeholder_token(PROMPT_TEMPLATE.replace('{}', ''))

    resolution = min(original_resolution)

    audio_driven_embedding = model.encode_audio(
        audio.to(model.device),
        placeholder_tokens,
        TEXT_POS_AT_PROMPT,
        PROMPT_LENGTH
    )

    v_seg = []
    for i in range(frames.shape[0]):
        out_dict = model(
            frames[i].unsqueeze(0).to(DEVICE),
            resolution=INPUT_RESOLUTION,
            pred_emb=audio_driven_embedding
        )

        seg = ((out_dict['positive'].squeeze().cpu().numpy()) * 255).astype(np.uint8)

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

    print(f'{audio.shape=}')

    video_file_name = video_file.split('/')[-1]
    print(video_file_name)
    if any(video_file_name.startswith(cat) for cat in ['original', 'silence', 'noise', 'offscreen']):
        likely_output_path = os.path.join(
            'data/examples/v_seg',
            video_file_name.removesuffix('.mp4') + '_' + '_'.join([model_name, model_version, 'v_seg.npy'])
        )
        if os.path.exists(likely_output_path):
            v_seg = np.load(likely_output_path)
        else:
            v_seg = forward_video(frames, audio, model_name, model_version, original_resolution)
            np.save(likely_output_path, v_seg)
    else:
        v_seg = forward_video(frames, audio, model_name, model_version, original_resolution)

    # Store in state
    state['video_seg'] = v_seg
    state['video_resolution'] = original_resolution
    state['video_audio_path'] = audio_path
    state['video_fps'] = fps

    # Create overlaid image
    v_overlaid = []
    v_heatmap_mask = []
    for i in range(v_seg.shape[0]):
        seg = v_seg[i]
        heatmap = draw_heatmap(seg, original_resolution)
        v_overlaid.append(draw_overlaid(np.array(original_frames[i]), heatmap))

        # Apply threshold
        seg_thresholded = apply_threshold_to_segmentation(seg, threshold)
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


# ============================= EXAMPLE VIDEO MANAGEMENT ============================

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
    examples_dir = 'data/examples'
    examples_map = ['original', 'silence', 'noise', 'offscreen']

    examples_dict = {}

    for category in examples_map:
        examples_dict[category] = []

        if os.path.exists(examples_dir):
            video_files = sorted([
                os.path.join(examples_dir, f) for f in os.listdir(examples_dir) if f.startswith(category)
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


# ========================================== APPLICATION ==========================================

title = "Audio-Grounded Contrastive Learning"

choices_models = ['ACL', 'ADCL']
choices_versions = {
    'ACL': [
        ('Baseline', 'baseline'),
        ('Retrained baseline', 'v1'),
        ('SaN', 'v2')
    ]
}
choices_models_init = choices_models[0]


def update_versions(model_name):
    return gr.Dropdown(
        choices=choices_versions[model_name],
        value=choices_versions[model_name][0]
    )


with gr.Blocks() as demo:
    gr.Markdown("Start typing below and then click **Run** to see the output.")

    # Initialize session state per client
    session_state = gr.State(delete_callback=cleanup_session)
    demo.load(fn=create_session, outputs=session_state)

    with gr.Row():
        model_name_in = gr.Dropdown(choices=choices_models, label="Model")
        model_version_name_in = gr.Dropdown(choices=choices_versions[choices_models_init], label="Version")
        model_name_in.change(fn=update_versions, inputs=model_name_in, outputs=model_version_name_in)

    with gr.Tabs():
        # ============= VIDEO TAB =============
        with gr.TabItem("Video"):
            gr.Markdown("### Video Input & Examples")

            # Load example videos
            example_videos_dict = load_example_videos()
            example_videos_count = sum(len(v) for v in example_videos_dict.values())

            with gr.Row():
                video_in = gr.Video(label="Video Input")

            # Display examples if available
            if example_videos_count > 0:
                gr.Markdown(f"**Available examples:** {' | '.join([f'{cat} ({len(vids)})' for cat, vids in example_videos_dict.items() if vids])}")
                example_videos = organize_examples_for_gradio(example_videos_dict)

                if example_videos:
                    gr.Examples(
                        examples=example_videos,
                        inputs=[video_in],
                        label="Click to load an example video"
                    )

            btn_video = gr.Button("Run")

            with gr.Row():
                v_heatmap_out = gr.Video(label="Heatmap (Grayscale)")
                v_overlaid_out = gr.Video(label="Overlaid with Original")

            with gr.Row():
                threshold_slider_video = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    label="Threshold",
                    info="Lower = more sensitive, Higher = less sensitive",
                    interactive=False
                )

            btn_video.click(
                fn=submit_video,
                inputs=[video_in, model_name_in, model_version_name_in, threshold_slider_video, session_state],
                outputs=[v_heatmap_out, v_overlaid_out, session_state]
            ).then(
                fn=lambda: gr.update(interactive=True),  # Enable slider after results
                outputs=threshold_slider_video
            )

            threshold_slider_video.change(
                fn=update_threshold_video,
                inputs=[threshold_slider_video, session_state],
                outputs=v_heatmap_out
            )

        # ============= IMAGE + AUDIO TAB =============
        with gr.TabItem("Image + Audio"):
            with gr.Row():
                image_in = gr.Image(type='pil', label="Image Input")
                audio_in = gr.Audio(label="Audio Input")

            btn = gr.Button("Run")

            with gr.Row():
                heatmap_out = gr.Image(type='pil', label="Heatmap (Grayscale)")
                overlaid_out = gr.Image(type='pil', label="Overlaid with Original")

            with gr.Row():
                threshold_slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    label="Threshold",
                    info="Lower = more sensitive, Higher = less sensitive",
                    interactive=False
                )

            btn.click(
                fn=submit,
                inputs=[image_in, audio_in, model_name_in, model_version_name_in, threshold_slider, session_state],
                outputs=[heatmap_out, overlaid_out, session_state]
            ).then(
                fn=lambda: gr.update(interactive=True),  # Enable slider after results
                outputs=threshold_slider
            )

            threshold_slider.change(
                fn=update_threshold,
                inputs=[threshold_slider, session_state],
                outputs=heatmap_out
            )


demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
