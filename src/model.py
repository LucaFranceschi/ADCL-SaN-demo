from importlib import import_module
from typing import cast, Optional

import torchaudio

from .constants import *

MODEL_REGISTRY = {
'baseline': {'display_name': 'ACL-SSL Baseline',
             'model_classname': 'ACL',
             'group': 'ACL-SaN',
             'weights_path': 'ACL_ViT16_test_best_param/Param_best.pth',
             'config_file_path': 'ACL_ViT16.yaml',
             'univ_thresh': 0.87},
'ACL-SaN_v1_B16': {'display_name': 'ACL-SaN v1',
                   'model_classname': 'ACL',
                   'group': 'ACL-SaN',
                   'weights_path': 'ACL-SaN_v1_B16_E17.pth',
                   'config_file_path': 'ACL_ViT16.yaml',
                   'univ_thresh': 0.92},
'ACL-SaN_v1_B32': {'display_name': 'ACL-SaN v1 (B32)',
                   'model_classname': 'ACL',
                   'group': 'ACL-SaN',
                   'weights_path': 'ACL-SaN_v1_B32_E19.pth',
                   'config_file_path': 'ACL_ViT16.yaml',
                   'univ_thresh': 0.93},
'ACL-SaN_v2_B16': {'display_name': 'ACL-SaN v2',
                   'model_classname': 'ACL',
                   'group': 'ACL-SaN',
                   'weights_path': 'ACL-SaN_v2_B16_E16.pth',
                   'config_file_path': 'ACL_ViT16.yaml',
                   'univ_thresh': 0.883},
'ACL-SaN_v3_B16': {'display_name': 'ACL-SaN v3',
                   'model_classname': 'ACL',
                   'group': 'ACL-SaN',
                   'weights_path': 'ACL-SaN_v3_B16_E15.pth',
                   'config_file_path': 'ACL_ViT16.yaml',
                   'univ_thresh': 0.876},
'ACL-SaN_v4_B16': {'display_name': 'ACL-SaN v4',
                   'model_classname': 'ACL',
                   'group': 'ACL-SaN',
                   'weights_path': 'ACL-SaN_v4_B16_E18.pth',
                   'config_file_path': 'ACL_ViT16.yaml',
                   'univ_thresh': 0.875},
'ACL-SaN_v5_B16': {'display_name': 'ACL-SaN v5',
                   'model_classname': 'ACL',
                   'group': 'ACL-SaN',
                   'weights_path': 'ACL-SaN_v5_B16_E16.pth',
                   'config_file_path': 'ACL_ViT16.yaml',
                   'univ_thresh': 0.613},
'ADCL_vA_B16': {'display_name': 'ADCL vA',
                'model_classname': 'ADCL',
                'group': 'ADCL',
                'weights_path': 'ACL-SaN_v1_B16_E17.pth',
                'config_file_path': 'ADCL_ViT16.yaml',
                'univ_thresh': 0.642},
'ADCL_vB_B16': {'display_name': 'ADCL vB',
                'model_classname': 'ADCL',
                'group': 'ADCL',
                'weights_path': 'ADCL_vB_B16_E18.pth',
                'config_file_path': 'ADCL_ViT16.yaml',
                'univ_thresh': 0.384},
'ADCL_vC_B16': {'display_name': 'ADCL vC',
                'model_classname': 'ADCL',
                'group': 'ADCL',
                'weights_path': 'ADCL_vC_B16_E17.pth',
                'config_file_path': 'ADCL_ViT16-v2.yaml',
                'univ_thresh': 0.842}
}

# ========================================= MODEL WRAPPER ==========================================

if "MODELS" not in globals():
    global MODELS
    MODELS = {}

def cleanup():
    global MODELS
    MODELS = {}

def multiton(cls):
    global MODELS
    def getinstance(name, **kwargs):
        if name not in MODELS:
            MODELS[name] = cls(name, **kwargs)
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
        univ_thresh: float|None = None,
        group: float|None = None,
    ) -> None:
        self.model_version = model_version
        self.display_name = display_name
        self.model_classname = model_classname
        self.weights_path = WEIGHTS_PATH.format(weights_path)
        self.config_file_path = CONFIGS_PATH.format(config_file_path)
        self.univ_thresh = univ_thresh
        self.group = group

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

    def embed_audio(self, audio: torch.Tensor) -> torch.Tensor:
        assert(self.model != None)
        placeholder_tokens = self.model.get_placeholder_token(PROMPT_TEMPLATE.replace('{}', ''))

        return self.model.encode_audio(
            audio.to(self.model.device),
            placeholder_tokens,
            TEXT_POS_AT_PROMPT,
            PROMPT_LENGTH
        )

# Instantiate all models from the registry
for key, cfg in MODEL_REGISTRY.items():
    Model(key, **{k: v for k, v in cfg.items()})

# ======================================= HELPER FUNCTIONS =========================================

def load_audio(audio_file: tuple[int, np.ndarray] | str | torch.Tensor) -> torch.Tensor:
    if type(audio_file) == torch.Tensor:
        return audio_file
    elif type(audio_file) == str:
        audio, sr = torchaudio.load(audio_file)  # type: ignore

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            audio = resampler(audio)

        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)

        return audio
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

        return audio

'''
Backported from TorchAudio (torchaudio.functional.add_noise)
Source: https://github.com/pytorch/audio/blob/e284e58c83f69c95a7f4a8a7d402f6c27ef56f5d/src/torchaudio/functional/functional.py#L2317

Copyright (c) 2017 Facebook Inc. (Soumith Chintala)
Licensed under the BSD 2-Clause License.
Reason: Version compatibility for torchaudio==0.13.0
'''
def add_noise(
    waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor, lengths: Optional[torch.Tensor] = None
) -> torch.Tensor:
    r'''Scales and adds noise to waveform per signal-to-noise ratio.

    Specifically, for each pair of waveform vector :math:`x \in \mathbb{R}^L` and noise vector
    :math:`n \in \mathbb{R}^L`, the function computes output :math:`y` as

    .. math::
        y = x + a n \, \text{,}

    where

    .. math::
        a = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}}{10}} } \, \text{,}

    with :math:`\text{SNR}` being the desired signal-to-noise ratio between :math:`x` and :math:`n`, in dB.

    Note that this function broadcasts singleton leading dimensions in its inputs in a manner that is
    consistent with the above formulae and PyTorch's broadcasting semantics.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
        noise (torch.Tensor): Noise, with shape `(..., L)` (same shape as ``waveform``).
        snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
        lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform`` and ``noise``, with shape
            `(...,)` (leading dimensions must match those of ``waveform``). If ``None``, all elements in ``waveform``
            and ``noise`` are treated as valid. (Default: ``None``)

    Returns:
        torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
        (same shape as ``waveform``).
    '''

    if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim and (lengths is None or lengths.ndim == snr.ndim)):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if L != noise.size(-1):
        raise ValueError(f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)}).")

    # compute scale
    if lengths is not None:
        mask = torch.arange(0, L, device=lengths.device).expand(waveform.shape) < lengths.unsqueeze(
            -1
        )  # (*, L) < (*, 1) = (*, L)
        masked_waveform = waveform * mask
        masked_noise = noise * mask
    else:
        masked_waveform = waveform
        masked_noise = noise

    energy_signal = torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2  # (*,)
    energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise  # (*, L)
