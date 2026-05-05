#!/usr/bin/env python3
"""
Video Audio Transformation Script

This script duplicates videos from data/examples/original/ with modified audio:
- silence/: Same frames with zero audio
- noise/: Same frames with Gaussian noise audio
- offscreen/: Same frames with swapped audio from another video (round-robin)

Usage:
    python transform_video_audio.py
"""

import os
import cv2
import torch
import torchaudio
import numpy as np
import subprocess
import uuid
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm


# =========================================== CONSTANTS ===========================================

SAMPLE_RATE = 16000
INPUT_DIR = 'data/examples/original'
OUTPUT_DIRS = {
    'silence': 'data/examples/silence',
    'noise': 'data/examples/noise',
    'offscreen': 'data/examples/offscreen'
}

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')


# =========================================== UTILITY FUNCTIONS ===================================

def add_noise(
    waveform: torch.Tensor,
    noise: torch.Tensor,
    snr: torch.Tensor,
    lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Scales and adds noise to waveform per signal-to-noise ratio.
    Backported from TorchAudio functional.

    Args:
        waveform: Input waveform with shape (..., L)
        noise: Noise tensor with same shape as waveform
        snr: Signal-to-noise ratio in dB
        lengths: Valid lengths of signals (optional)

    Returns:
        torch.Tensor: Waveform with added noise
    """
    # Compute power of waveform and noise
    power_waveform = (waveform ** 2).mean(dim=-1)
    power_noise = (noise ** 2).mean(dim=-1)

    # Avoid division by zero
    power_noise = torch.where(
        power_noise == 0,
        torch.ones_like(power_noise),
        power_noise
    )

    # Calculate scaling factor
    snr_db = snr.reshape(power_waveform.shape)
    snr_linear = 10.0 ** (snr_db / 10.0)

    # Compute scaling factor for noise
    scale = torch.sqrt(power_waveform / power_noise / snr_linear)

    # Reshape scale for broadcasting
    while len(scale.shape) < len(noise.shape):
        scale = scale.unsqueeze(-1)

    # Add scaled noise to waveform
    return waveform + scale * noise


class AddRandomNoise(torch.nn.Module):
    """Add Gaussian noise to audio with SNR control"""

    def __init__(self, snr: float = None):
        """
        Args:
            snr: Signal-to-noise ratio in dB. If None, uses high value (minimal noise)
        """
        super().__init__()
        if snr is not None:
            self.snr = torch.Tensor([snr])
        else:
            self.snr = torch.Tensor([1000.0])  # High value = no noise

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: Input audio tensor with shape (L,) or (C, L)

        Returns:
            torch.Tensor: Audio with added noise
        """
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)

        noise = torch.clip(torch.randn(waveform.shape), min=-1., max=1.)
        noisy_waveform = add_noise(waveform, noise, self.snr, None)

        return noisy_waveform.squeeze(0) if noisy_waveform.shape[0] == 1 else noisy_waveform


# =========================================== VIDEO/AUDIO FUNCTIONS ==============================

def extract_video_frames(
    video_path: str,
    resolution: Optional[Tuple[int, int]] = None
) -> Tuple[List[np.ndarray], Tuple[int, int], float]:
    """
    Extract frames from video file.

    Args:
        video_path: Path to video file
        resolution: Target resolution (width, height). If None, uses original

    Returns:
        Tuple of (frames_list, original_resolution, fps)
    """
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    original_resolution = (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    fps = video.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = video.read()
        if not ret:
            break

        if resolution and resolution != original_resolution:
            frame = cv2.resize(frame, resolution)

        frames.append(frame)

    video.release()

    return frames, original_resolution, fps


def extract_audio_from_video(
    video_path: str,
    output_audio_path: str,
    sample_rate: int = SAMPLE_RATE
) -> str:
    """
    Extract audio from video using ffmpeg.

    Args:
        video_path: Path to video file
        output_audio_path: Path to save extracted audio
        sample_rate: Target sample rate in Hz

    Returns:
        Path to extracted audio file
    """
    try:
        subprocess.run(
            [
                'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(sample_rate), '-ac', '1', '-y', output_audio_path
            ],
            capture_output=True,
            check=True,
            timeout=300
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error for {video_path}: {e.stderr.decode()}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install ffmpeg.")

    return output_audio_path


def load_audio(audio_path: str, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Load audio from file and resample to target sample rate.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate in Hz

    Returns:
        torch.Tensor: Audio tensor with shape (num_samples,)
    """
    audio, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        audio = resampler(audio)

    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)

    return audio.squeeze(0)


def save_audio(audio: torch.Tensor, output_path: str, sample_rate: int = SAMPLE_RATE) -> str:
    """
    Save audio tensor to file.

    Args:
        audio: torch.Tensor with shape (num_samples,)
        output_path: Path to save audio
        sample_rate: Sample rate in Hz

    Returns:
        Path to saved audio file
    """
    # Ensure audio is in correct format
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    # Clip values to valid range
    audio = torch.clamp(audio, -1.0, 1.0)

    torchaudio.save(output_path, audio, sample_rate)
    return output_path


def save_video(
    frames: List[np.ndarray],
    audio_path: str,
    output_video_path: str,
    fps: float = 30.0
) -> str:
    """
    Save video frames with audio using ffmpeg.

    Args:
        frames: List of frames (numpy arrays)
        audio_path: Path to audio file
        output_video_path: Path to save output video
        fps: Frames per second

    Returns:
        Path to saved video file
    """
    if not frames:
        raise ValueError("No frames to save")

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Create temporary AVI file
    temp_video_path = os.path.join(
        os.path.dirname(output_video_path),
        f"temp_{uuid.uuid4()}.avi"
    )

    # Write frames to AVI
    fourcc = cv2.VideoWriter.fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise RuntimeError(f"Could not create video writer at {temp_video_path}")

    for frame in frames:
        # Ensure frame is in BGR format
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 3:  # RGB
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            pass

        video_writer.write(frame)

    video_writer.release()

    # Merge video and audio using ffmpeg
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    try:
        subprocess.run(
            [
                'ffmpeg', '-y', '-i', temp_video_path, '-i', audio_path,
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
                '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0',
                output_video_path
            ],
            capture_output=True,
            check=True,
            timeout=600
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error during merge: {e.stderr.decode()}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

    return output_video_path


# =========================================== AUDIO MODIFICATION FUNCTIONS =======================

def create_silence_audio(duration_samples: int, sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """
    Create silence (zero) audio.

    Args:
        duration_samples: Number of samples
        sample_rate: Sample rate in Hz (for reference)

    Returns:
        torch.Tensor: Silent audio
    """
    return torch.zeros(duration_samples)


def create_noise_audio(
    duration_samples: int,
    snr_db: float = 10.0,
    sample_rate: int = SAMPLE_RATE
) -> torch.Tensor:
    """
    Create Gaussian noise audio.

    Args:
        duration_samples: Number of samples
        snr_db: Signal-to-noise ratio in dB (for reference, not used for pure noise)
        sample_rate: Sample rate in Hz (for reference)

    Returns:
        torch.Tensor: Noise audio
    """
    # Generate pure Gaussian noise
    noise = torch.randn(duration_samples)

    # Normalize to reasonable amplitude
    noise = noise / (torch.std(noise) + 1e-8)
    noise = torch.clamp(noise * 0.1, -1.0, 1.0)  # Scale to reasonable amplitude

    return noise


def swap_audio_round_robin(
    audio_list: List[torch.Tensor],
    video_indices: List[int]
) -> List[torch.Tensor]:
    """
    Swap audio between videos in round-robin fashion.
    Video i gets audio from video (i+1) % n_videos.

    Args:
        audio_list: List of audio tensors
        video_indices: Original indices of videos

    Returns:
        List[torch.Tensor]: Swapped audio list
    """
    n_videos = len(audio_list)
    swapped_audio = [None] * n_videos

    for i in range(n_videos):
        # Video i gets audio from video (i+1) % n_videos
        source_idx = (i + 1) % n_videos

        # Pad/trim audio to match duration if needed
        target_duration = audio_list[i].shape[0]
        source_audio = audio_list[source_idx]

        if source_audio.shape[0] < target_duration:
            # Pad with silence
            padding = target_duration - source_audio.shape[0]
            source_audio = torch.cat([
                source_audio,
                torch.zeros(padding)
            ])
        elif source_audio.shape[0] > target_duration:
            # Trim
            source_audio = source_audio[:target_duration]

        swapped_audio[i] = source_audio

    return swapped_audio


# =========================================== MAIN PROCESSING FUNCTION ===========================

def process_videos():
    """
    Main function to process all videos in input directory.
    Creates modified versions with silence, noise, and swapped audio.
    """
    # Create output directories
    for output_dir in OUTPUT_DIRS.values():
        os.makedirs(output_dir, exist_ok=True)

    # Find all video files
    input_path = Path(INPUT_DIR)
    video_files = sorted([
        f for f in input_path.glob('*')
        if f.is_file() and f.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']
    ])

    if not video_files:
        print(f"No video files found in {INPUT_DIR}")
        return

    print(f"Found {len(video_files)} video(s) to process")
    print(f"Output directories:")
    for mode, path in OUTPUT_DIRS.items():
        print(f"  - {mode}: {path}")
    print()

    # Load all videos and audio
    print("Loading videos and audio...")
    video_data = []
    audio_list = []

    for idx, video_path in enumerate(tqdm(video_files, desc="Loading videos")):
        try:
            # Extract frames and metadata
            frames, original_resolution, fps = extract_video_frames(str(video_path))

            # Extract audio
            temp_audio_path = f"/tmp/temp_audio_{uuid.uuid4()}.wav"
            extract_audio_from_video(str(video_path), temp_audio_path)
            audio = load_audio(temp_audio_path)

            video_data.append({
                'path': video_path,
                'filename': video_path.stem,
                'frames': frames,
                'resolution': original_resolution,
                'fps': fps
            })
            audio_list.append(audio)

            # Clean up temporary audio
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue

    if not video_data:
        print("No videos were successfully loaded")
        return

    print(f"Successfully loaded {len(video_data)} video(s)\n")

    # Get duration for all audio files (in samples)
    audio_durations = [audio.shape[0] for audio in audio_list]

    # Process each video
    print("Processing videos with audio modifications...")

    for idx, (data, original_audio) in enumerate(tqdm(
        zip(video_data, audio_list),
        total=len(video_data),
        desc="Processing"
    )):
        filename = data['filename']
        output_ext = '.mp4'

        # Create temporary directory for intermediate files
        temp_dir = f"/tmp/video_processing_{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # ============= SILENCE MODE =============
            silence_audio = create_silence_audio(original_audio.shape[0])
            temp_silence_audio = os.path.join(temp_dir, 'silence_audio.wav')
            save_audio(silence_audio, temp_silence_audio)

            silence_output = os.path.join(
                OUTPUT_DIRS['silence'],
                f"{filename}{output_ext}"
            )
            save_video(data['frames'], temp_silence_audio, silence_output, data['fps'])

            # ============= NOISE MODE =============
            noise_audio = create_noise_audio(original_audio.shape[0])
            temp_noise_audio = os.path.join(temp_dir, 'noise_audio.wav')
            save_audio(noise_audio, temp_noise_audio)

            noise_output = os.path.join(
                OUTPUT_DIRS['noise'],
                f"{filename}{output_ext}"
            )
            save_video(data['frames'], temp_noise_audio, noise_output, data['fps'])

            # ============= OFFSCREEN MODE (SWAPPED AUDIO) =============
            # Prepare audio list for swapping
            swapped_audios = swap_audio_round_robin(audio_list, list(range(len(audio_list))))
            swapped_audio = swapped_audios[idx]

            temp_swapped_audio = os.path.join(temp_dir, 'swapped_audio.wav')
            save_audio(swapped_audio, temp_swapped_audio)

            offscreen_output = os.path.join(
                OUTPUT_DIRS['offscreen'],
                f"{filename}{output_ext}"
            )
            save_video(data['frames'], temp_swapped_audio, offscreen_output, data['fps'])

        except Exception as e:
            print(f"Error processing {filename}: {e}")

        finally:
            # Clean up temporary files
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

    print("\n✓ Video processing complete!")
    print(f"\nOutput summary:")
    for mode, output_dir in OUTPUT_DIRS.items():
        output_count = len(list(Path(output_dir).glob('*')))
        print(f"  - {mode}: {output_count} video(s)")


# =========================================== ENTRY POINT ========================================

if __name__ == '__main__':
    print("=" * 60)
    print("Video Audio Transformation Script")
    print("=" * 60)
    print()

    process_videos()