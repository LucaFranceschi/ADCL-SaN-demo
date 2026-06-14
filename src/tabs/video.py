import os, uuid, subprocess

import torch
import torchaudio
import numpy as np
import cv2

from PIL import Image
from PIL.Image import Image as PImage
from torchvision import transforms as vt
from typing import cast
from gradio import Progress, skip

from ..model import Model
from ..constants import *
from ..session import SessionState

from utils.viz import draw_overlaid, draw_heatmap

# ======================================= VIDEO TAB FUNCTIONS ======================================

@torch.no_grad()
def forward_video(
    frames: torch.Tensor,
    audio: torch.Tensor,
    model_version: str,
    progress: Progress
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
    progress(0, desc="Starting")
    v_seg = []
    for i in progress.tqdm(range(frames.shape[0]), desc=f'Processing video...'):
        out_dict = module(
            frames[i].unsqueeze(0).to(DEVICE), #type: ignore
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
    state: SessionState,
    progress=Progress()
) -> tuple[str, str, SessionState]:
    """Submit video and return heatmap and overlaid visualization"""
    progress(0, desc=f"Loading input video...")

    # Extract video frames
    video = cv2.VideoCapture(video_file)
    original_resolution = (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )

    fps = int(video.get(cv2.CAP_PROP_FPS))

    image_transform = vt.Compose([
        vt.Resize((INPUT_RESOLUTION, INPUT_RESOLUTION), vt.InterpolationMode.BICUBIC),
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
            VIDEO_EXAMPLES_PATH,
            'v_seg',
            video_file_name.removesuffix('.mp4') + '_' + '_'.join([model_version, 'v_seg.npy'])
        )
        if os.path.exists(likely_output_path):
            progress(0.5, desc=f"Loading cached example...")
            v_seg = np.load(likely_output_path)
        else:
            progress(0, desc=f"Example not cached. This might take a while...")
            v_seg = forward_video(frames, audio, model_version, progress)
            np.save(likely_output_path, v_seg)
    else:
        progress(0, desc=f"This might take a while...")
        v_seg = forward_video(frames, audio, model_version, progress)

    # Store in state
    state['video_seg'] = v_seg
    state['video_resolution'] = original_resolution
    state['video_audio_path'] = audio_path
    state['video_fps'] = fps

    model = Model(model_version)
    assert(model.univ_thresh != None)

    progress(1, desc=f"Showing output...")

    # Create overlaid image
    v_overlaid = []
    v_heatmap_mask = []
    for i in range(v_seg.shape[0]):
        seg = v_seg[i]
        heatmap = draw_heatmap(seg, original_resolution)
        v_overlaid.append(draw_overlaid(np.array(original_frames[i]), heatmap))

        # Apply threshold
        seg_thresholded = apply_threshold_to_segmentation(seg, model.univ_thresh)
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

# @gr.cache
def update_threshold_video(
    thresh_type: str,
    thresh_value: float,
    model_version: str,
    state: SessionState
) -> str:
    """Update threshold for video segmentation"""
    if 'video_seg' not in state or \
        'video_resolution' not in state or \
        'video_audio_path' not in state or \
        'video_fps' not in state:
        return skip() # type: ignore

    used_threshold = 0.5
    if thresh_type == 'custom':
        used_threshold = thresh_value
    elif hasattr(Model(model_version), thresh_type):
        used_threshold = getattr(Model(model_version), thresh_type)

    v_heatmap_mask = []
    for i in range(state['video_seg'].shape[0]):
        if thresh_type == 'top50p':
            used_threshold = cast(float, np.median(state['video_seg'][i]))
        seg_thresholded = apply_threshold_to_segmentation(state['video_seg'][i], used_threshold)
        v_heatmap_mask.append(draw_heatmap(seg_thresholded, state['video_resolution']))

    heatmap_mask = save_video(
        v_heatmap_mask,
        state['video_audio_path'],
        os.path.join(state['session_dir'], 'video_mask.mp4'),
        state['video_resolution'],
        state['video_fps']
    )

    return heatmap_mask

def load_example_videos() -> dict[str, list[str]]:
    examples_map = ['original', 'silence', 'noise', 'offscreen']

    examples_dict = {}

    for category in examples_map:
        examples_dict[category] = []

        if os.path.exists(VIDEO_EXAMPLES_PATH):
            video_files = sorted([
                os.path.join(VIDEO_EXAMPLES_PATH, f) for f in os.listdir(VIDEO_EXAMPLES_PATH) if f.startswith(category)
            ])
            examples_dict[category] = video_files

    return examples_dict

def organize_examples_for_gradio(examples_dict: dict[str, list[str]]) -> list[list[str]]:
    examples_map = ['original', 'silence', 'noise', 'offscreen']
    examples_list = []

    for category in examples_map:
        examples_list += examples_dict[category]

    return examples_list