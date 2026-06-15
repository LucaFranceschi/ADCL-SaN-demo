import torch
import numpy as np

from utils.util import get_prompt_template

# ========================================== ENV SETTINGS ==========================================

WEIGHTS_PATH = 'data/models/{}'
CONFIGS_PATH = 'config/model/{}'
PT_MODELS_PATH = 'data/pretrain'
VIDEO_EXAMPLES_PATH = 'data/examples/videos'
FRAMES_EXAMPLES_PATH = 'data/examples/frames'
AUDIOS_EXAMPLES_PATH = 'data/examples/audios'
MEDIA_DIR = 'media'

EXAMPLES = {
    'BNfeHeas6hA_000076': 'roar',
    'ixscoaWEEnQ_000104': 'chew',
    '2Bljhdt61Y4_000038': 'bassoon'
}

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda', torch.cuda.current_device()) if USE_CUDA else torch.device('cpu')
print(f'Device: {DEVICE} is used\n')

# =========================================== CONSTANTS ============================================

INPUT_RESOLUTION = 352
SAMPLE_RATE = 16000
VIDEO_AUDIO_WINDOW = 3
PROMPT_TEMPLATE, TEXT_POS_AT_PROMPT, PROMPT_LENGTH = get_prompt_template()

# =========================================== FUNCTIONS ============================================

def apply_threshold_to_segmentation(seg: np.ndarray, threshold: float) -> np.ndarray:
    """Apply threshold to segmentation map"""
    seg_thresholded = np.where(seg >= threshold*255, 0, 255).astype(np.uint8)
    return seg_thresholded
