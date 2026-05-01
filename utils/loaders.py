import io

import torch
import torchaudio
from torchvision import transforms as vt

from fastapi import File, UploadFile
from PIL import Image

SAMPLE_RATE = 16000

async def get_audio(audio_file: UploadFile = File(...)) -> torch.Tensor:
    audio_bytes = await audio_file.read()
    audio, sr = torchaudio.load(io.BytesIO(audio_bytes))

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio_file = resampler(audio)

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0)

    # simulate batch dimension
    audio = audio.unsqueeze(0)

    return audio

async def get_image(image_file: UploadFile = File(...)) -> tuple[Image.Image, torch.Tensor]:
    image_bytes = await image_file.read()
    original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    resolution = min(original_image.width, original_image.height)
    image_transform = vt.Compose([
        vt.Resize((resolution, resolution), vt.InterpolationMode.BICUBIC),
        vt.ToTensor(),
        vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
    ])
    image = image_transform(original_image)

    # simulate batch dimension
    image = image.unsqueeze(0)

    return original_image, image
