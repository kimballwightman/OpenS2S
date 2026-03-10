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
        # ============================================================
        # WAVLM CODE (ACTIVE) - WavLM audio encoder
        # ============================================================
        if isinstance(audio_encoder_config, dict):
            # Force WavLM as the only audio encoder
            if "model_type" not in audio_encoder_config or audio_encoder_config["model_type"] != "wavlm":
                logger.warning(f"Audio encoder model_type was '{audio_encoder_config.get('model_type', 'None')}', forcing to 'wavlm'")
                audio_encoder_config["model_type"] = "wavlm"

            # Set WavLM defaults
            audio_encoder_config.setdefault("hidden_size", 768)
            audio_encoder_config.setdefault("d_model", 1280)
            audio_encoder_config.setdefault("num_hidden_layers", 12)
            audio_encoder_config.setdefault("max_source_positions", 1500)
            audio_encoder_config.setdefault("num_attention_heads", 12)
            audio_encoder_config.setdefault("intermediate_size", 3072)
            audio_encoder_config.setdefault("activation_function", "gelu")
            audio_encoder_config.setdefault("dropout", 0.1)
            audio_encoder_config.setdefault("attention_dropout", 0.1)
        elif audio_encoder_config is None:
            logger.info("audio encoder config is None. Initializing with wavlm")
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
                "attention_dropout": 0.1
            }
        else:
            raise NotImplementedError
        # ============================================================

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

    @classmethod
    def from_separate_repos(cls, audio_encoder_repo, llm_repo, tts_repo, adapter_repo):
        """
        Load configs from 4 separate HuggingFace repos.

        Each repo should have its own config file:
        - audio_encoder_repo: Contains audio_encoder_config.json
        - llm_repo: Contains llm_config.json
        - tts_repo: Contains tts_lm_config.json
        - adapter_repo: Contains config.json with adapter settings

        Args:
            audio_encoder_repo: Path or repo ID for audio encoder
            llm_repo: Path or repo ID for LLM
            tts_repo: Path or repo ID for TTS
            adapter_repo: Path or repo ID for adapters

        Returns:
            OmniSpeechConfig instance with configs loaded from separate repos
        """
        from huggingface_hub import hf_hub_download
        import json
        import os

        # Load audio encoder config
        if os.path.exists(os.path.join(audio_encoder_repo, "audio_encoder_config.json")):
            # Local path
            audio_config_path = os.path.join(audio_encoder_repo, "audio_encoder_config.json")
        else:
            # HF repo
            audio_config_path = hf_hub_download(
                repo_id=audio_encoder_repo,
                filename="audio_encoder_config.json"
            )
        with open(audio_config_path) as f:
            audio_encoder_config = json.load(f)

        # Load LLM config
        if os.path.exists(os.path.join(llm_repo, "llm_config.json")):
            llm_config_path = os.path.join(llm_repo, "llm_config.json")
        else:
            llm_config_path = hf_hub_download(
                repo_id=llm_repo,
                filename="llm_config.json"
            )
        with open(llm_config_path) as f:
            llm_config = json.load(f)

        # Load TTS config
        if os.path.exists(os.path.join(tts_repo, "tts_lm_config.json")):
            tts_config_path = os.path.join(tts_repo, "tts_lm_config.json")
        else:
            tts_config_path = hf_hub_download(
                repo_id=tts_repo,
                filename="tts_lm_config.json"
            )
        with open(tts_config_path) as f:
            tts_lm_config = json.load(f)

        # Load adapter config (use full config.json from adapters repo)
        adapter_config = cls.from_pretrained(adapter_repo)

        # Force WavLM
        audio_encoder_config["model_type"] = "wavlm"

        return cls(
            audio_encoder_config=audio_encoder_config,
            llm_config=llm_config,
            tts_lm_config=tts_lm_config,
            lora_config=adapter_config.lora_config,
            conv_kernel_sizes=adapter_config.conv_kernel_sizes,
            adapter_inner_dim=adapter_config.adapter_inner_dim,
            interleave_strategy=adapter_config.interleave_strategy
        )
