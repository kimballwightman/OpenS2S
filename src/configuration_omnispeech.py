"""BLAP config"""

from transformers import PretrainedConfig
from transformers import logging
from peft import LoraConfig

from transformers.models.auto import CONFIG_MAPPING

logger = logging.get_logger(__name__)

class OmniSpeechConfig(PretrainedConfig):
    def __init__(
        self, 
        audio_encoder_config=None,
        llm_config=None,
        tts_lm_config=None,
        lora_config=None,
        conv_kernel_sizes="5,5",
        adapter_inner_dim=512,
        interleave_strategy="1:2",
        **kwargs
    ):
        # Force WavLM as the only audio encoder (no fallbacks)
        if isinstance(audio_encoder_config, dict):
            # Ensure model_type is wavlm
            if "model_type" not in audio_encoder_config or audio_encoder_config["model_type"] != "wavlm":
                logger.warning(f"Audio encoder model_type was '{audio_encoder_config.get('model_type', 'None')}', forcing to 'wavlm'")
                audio_encoder_config["model_type"] = "wavlm"

            # Set WavLM defaults
            audio_encoder_config.setdefault("hidden_size", 768)  # WavLM internal dimension
            audio_encoder_config.setdefault("d_model", 1280)  # Output dimension after projection
            audio_encoder_config.setdefault("num_hidden_layers", 12)
            audio_encoder_config.setdefault("max_source_positions", 1500)
            audio_encoder_config.setdefault("num_attention_heads", 12)
            audio_encoder_config.setdefault("intermediate_size", 3072)
            audio_encoder_config.setdefault("activation_function", "gelu")
            audio_encoder_config.setdefault("dropout", 0.1)
            audio_encoder_config.setdefault("attention_dropout", 0.1)

        elif audio_encoder_config is None:
            logger.info("audio encoder config is None. Initializing with wavlm (hardcoded)")
            audio_encoder_config = {
                "model_type": "wavlm",
                "hidden_size": 768,
                "d_model": 1280,
                "num_hidden_layers": 12,
                "max_source_positions": 1500,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "activation_function": "gelu",
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "activation_dropout": 0.0,
                "layerdrop": 0.0,
                "init_std": 0.02,
                "scale_embedding": False
            }
        else:
            raise NotImplementedError("audio_encoder_config must be dict or None")


        if isinstance(llm_config, dict):
            if "model_type" in llm_config:
                pass
            else:
                logger.info("llm config is None. Initializing with qwen3")
                llm_config["model_type"] = "qwen3"
        elif llm_config is None:
            logger.info("llm config is None. Initializing with qwen3")
            llm_config = {
                "model_type": "qwen3"
            }
        else:
            raise NotImplementedError
        
        if isinstance(tts_lm_config, dict):
            if "model_type" in tts_lm_config:
                pass
            else:
                logger.info("tts lm config is None. Initializing with qwen3")
                tts_lm_config["model_type"] = "qwen3"
        elif tts_lm_config is None:
            logger.info("tts lm config is None. Initializing with qwen3")
            tts_lm_config = {
                "model_type": "qwen3"
            }

        self.audio_encoder_config = CONFIG_MAPPING[audio_encoder_config["model_type"]](**audio_encoder_config)
        self.llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
        self.tts_lm_config = CONFIG_MAPPING[tts_lm_config["model_type"]](**tts_lm_config)

        self.lora_config = lora_config

        self.conv_kernel_sizes = conv_kernel_sizes
        self.adapter_inner_dim = adapter_inner_dim
        self.interleave_strategy = interleave_strategy

        super().__init__(**kwargs)