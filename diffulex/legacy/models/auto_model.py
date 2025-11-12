from diffulex.legacy.config import Config
from diffulex.legacy.utils.loader import load_model
from diffulex.legacy.models.dream import DreamForDiffusionLM
from diffulex.legacy.models.qwen3 import Qwen3ForCausalLM
from diffulex.legacy.models.llada import LLaDAForDiffusionLM


class AutoModelLM:
    MODEL_MAPPING = {
        "qwen3": Qwen3ForCausalLM,
        "dream": DreamForDiffusionLM,
        "llada": LLaDAForDiffusionLM,
        "llada-1.5": None,
        "dream-on": None,
        "sdar": None,
    }
    @classmethod
    def from_config(cls, config: Config):
        model = cls.MODEL_MAPPING[config.model_name](config.hf_config)
        return load_model(model, config)