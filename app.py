import os
import torch
import torchaudio

import numpy as np
import gradio as gr

from PIL.Image import Image
from importlib import import_module
from torchvision import transforms as vt

# from modules.models import ACL, ADCL
from utils.util import get_prompt_template
from utils.viz import draw_overlaid, draw_heatmap

# =========================================== CONSTANTS ===========================================

INPUT_RESOLUTION = 352
SAMPLE_RATE = 16000

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

# =========================================== FUNCTIONS ===========================================

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

def submit(
    image_file: Image,
    audio_file: tuple[int, np.ndarray],
    # video: UploadFile = File(...),
    model_name: str,
    model_version: str,
):
    resolution = min(image_file.width, image_file.height)
    image_transform = vt.Compose([
        vt.Resize((resolution, resolution), vt.InterpolationMode.BICUBIC),
        vt.ToTensor(),
        vt.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),  # CLIP
    ])
    image = image_transform(image_file)

    # simulate batch dimension
    image = image.unsqueeze(0)
    print(f'{image.shape=}')

    original_resolution = image_file.size

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

    return forward(image, audio, model_name, model_version, original_resolution)

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

    with gr.Row():
        model_name_in = gr.Dropdown(choices=choices_models)
        model_version_name_in = gr.Dropdown(choices=choices_versions[choices_models_init])
        model_name_in.change(fn=update_versions, inputs=model_name_in, outputs=model_version_name_in)

        audio_in = gr.Audio()
        image_in = gr.Image(type='pil')
        image_out = gr.Image(type='pil')

    btn = gr.Button("Run")
    btn.click(fn=submit, inputs=[image_in, audio_in, model_name_in, model_version_name_in], outputs=image_out)

demo.launch(server_name="0.0.0.0", server_port=7860, debug=True)
