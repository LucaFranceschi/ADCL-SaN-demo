from importlib import import_module

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

    def embed_audio(self, audio) -> torch.Tensor:
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
