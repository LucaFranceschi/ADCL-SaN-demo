import os, io
import torch
import numpy as np
import base64

from PIL.Image import Image
from pathlib import Path
from importlib import import_module

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

# from modules.models import ACL, ADCL
from utils.util import get_prompt_template
from utils.loaders import get_image, get_audio
from utils.viz import draw_overlaid, draw_heatmap

# =========================================== CONSTANTS ===========================================

INPUT_RESOLUTION = 352
MODEL_PATH = 'data/pretrain'
CONFIG_FILE_TEMPLATE = 'config/model/{}_ViT16.yaml'
WEIGHTS_PATH = 'data/models'
WEIGHTS_SUBPATH = {
    'baseline': 'ACL_ViT16_test_best_param/Param_best.pth'
}

USE_CUDA = torch.cuda.is_available()

PROMPT_TEMPLATE, TEXT_POS_AT_PROMPT, PROMPT_LENGTH = get_prompt_template()

DEVICE = torch.device('cuda', torch.cuda.current_device()) if USE_CUDA else torch.device('cpu')
print(f'Device: {DEVICE} is used\n')

# ========================================== APPLICATION ==========================================

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_index():
    return FileResponse(Path('static/index.html'))

@app.post("/submit")
async def submit(
    image_file: UploadFile = File(...),
    audio_file: UploadFile = File(...),
    # video: UploadFile = File(...),
    model_name: str = Form(...),
    model_version: str = Form(...),
):
    original_image, image = await get_image(image_file)
    resolution = original_image.size

    audio = await get_audio(audio_file)

    heatmap = forward(image, audio, model_name, model_version, resolution)

    overlaid = draw_overlaid(original_image, heatmap)

    heatmap_buffer = io.BytesIO()
    heatmap.save(heatmap_buffer, format='JPEG')
    heatmap_buffer.seek(0)

    overlaid_buffer = io.BytesIO()
    overlaid.save(overlaid_buffer, format='JPEG')
    overlaid_buffer.seek(0)

    # return Response(content=(heatmap_buffer.getvalue(), overlaid_buffer.getvalue()), media_type='image/jpeg')
    return JSONResponse(content={
        'heatmap': base64.b64encode(heatmap_buffer.getvalue()).decode(),
        'overlaid': base64.b64encode(overlaid_buffer.getvalue()).decode()
    })

@torch.no_grad()
def forward(
    image: torch.Tensor,
    audio: torch.Tensor,
    model_name: str,
    model_version: str,
    resolution: tuple[int]
) -> Image:
    # perform a forward pass and return the heatmap as a grayscale image
    model = getattr(import_module('modules.models'), model_name)(
        CONFIG_FILE_TEMPLATE.format(model_name),
        DEVICE,
        MODEL_PATH
    )

    model.load(os.path.join(WEIGHTS_PATH, WEIGHTS_SUBPATH[model_version]))
    model.train(False)

    placeholder_tokens = model.get_placeholder_token(PROMPT_TEMPLATE.replace('{}', ''))

    min_resolution = min(resolution)

    audio_driven_embedding = model.encode_audio(
        audio.to(model.device),
        placeholder_tokens,
        TEXT_POS_AT_PROMPT,
        PROMPT_LENGTH
    )

    out_dict = model(image.to(DEVICE), resolution=min_resolution, pred_emb=audio_driven_embedding)

    seg = out_dict['positive']

    seg_image = ((seg.squeeze().cpu().numpy()) * 255).astype(np.uint8)

    return draw_heatmap(seg_image, resolution)
