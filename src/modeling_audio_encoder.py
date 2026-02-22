from typing import Optional, Tuple
from dataclasses import dataclass
import os
import glob

import torch
import safetensors
import torch
from transformers.utils import ModelOutput


from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder as HFWhisperEncoder
from transformers import Qwen2AudioEncoderConfig
from transformers import Qwen2AudioEncoder as HFQwen2AudioEncoder
from transformers import Wav2Vec2Config
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model


from .utils import length_to_attention_mask

@dataclass
class AudioEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    output_lengths: Optional[torch.LongTensor] = None


class WhisperEncoder(HFWhisperEncoder):
    """
    overwrite forward to support long audio
    overwrite from_pretrained to support split encoder parameters from pretrained WhisperModel
    """

    def from_pretrained(model_path):
        config = WhisperConfig.from_pretrained(model_path)

        model = WhisperEncoder(config)
        old_state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"))
        state_dict = {}
        for para_name in old_state_dict.keys():
            if "model.encoder." in para_name:
                new_name = para_name.replace("model.encoder.", "")
                state_dict[new_name] = old_state_dict[para_name]
        model.load_state_dict(state_dict)

        return model


    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        bz, hidden_dim, _ = input_features.size()
        input_features = input_features.transpose(1,2) # B x C x T -> B x T x C
        input_n_samples = self.max_source_positions * 2 # 3000

        input_lengths = attention_mask.sum(-1)
        segments = torch.ceil(input_lengths / input_n_samples).to(dtype=torch.long)
        input_features = input_features.contiguous().view(-1, input_n_samples, hidden_dim) # M x 3000 x C
        attention_mask = attention_mask.contiguous().view(-1, input_n_samples) # M x 3000
        ### filter empty input
        ### for example, when a1(15s) and a2(45s) in one batch
        ### the audios will be padded into 60s and then segmented into 30s like [a11, a12, a21, a22]
        ### we will skip the computation for a12
        select_index = length_to_attention_mask(segments).to(torch.bool).view(-1)
        input_features = input_features[select_index]
        attention_mask = attention_mask[select_index]
        input_features = input_features.transpose(1,2) # M x 3000 x C -> M x C x 3000

        output = super().forward(
            input_features,
            attention_mask,
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        )

        last_hidden_state = output.last_hidden_state # M x 1500 x C
        input_lengths = attention_mask.sum(-1) # M
        output_n_samples = last_hidden_state.size(1)
        ### concate the faetures of the same example
        output_last_hidden_state = last_hidden_state.new_zeros(
            bz, output_n_samples * segments.max().item(), last_hidden_state.size(2)
        )
        output_lengths = input_lengths.new_zeros(bz)
        idx = 0
        for i, l in enumerate(segments):
            output_last_hidden_state[i,:output_n_samples*l,:] = \
                last_hidden_state[idx:idx+l].contiguous().view(-1, last_hidden_state.size(2))
            output_lengths[i] = self._get_feat_extract_output_lengths(input_lengths[idx:idx+l]).sum()
            idx += l

        max_length = output_lengths.max()
        output_last_hidden_state = output_last_hidden_state[:,:max_length,:]

        return AudioEncoderOutput(
            last_hidden_state=output_last_hidden_state,
            hidden_states=None,
            attentions=None,
            output_lengths=output_lengths
        )


class Qwen2AudioEncoder(HFQwen2AudioEncoder):
    """
        overwrite forward to support long audio
        overwrite from_pretrained to support split encoder parameters from pretrained Qwen2Audio
        """

    def from_pretrained(model_path):
        config = Qwen2AudioEncoderConfig.from_pretrained(model_path)

        model = Qwen2AudioEncoder(config)
        state_dict = {}
        for path in glob.glob(os.path.join(model_path, "model*.safetensors")):
            with safetensors.safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "audio_tower" in key:
                        new_key = key.replace("audio_tower.", "")
                        state_dict[new_key] = f.get_tensor(key)
        model.load_state_dict(state_dict)

        return model


    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        bz, hidden_dim, _ = input_features.size()
        input_features = input_features.transpose(1,2) # B x C x T -> B x T x C
        input_n_samples = self.max_source_positions * 2 # 3000

        input_lengths = attention_mask.sum(-1)
        segments = torch.ceil(input_lengths / input_n_samples).to(dtype=torch.long)
        input_features = input_features.contiguous().view(-1, input_n_samples, hidden_dim) # M x 3000 x C
        attention_mask = attention_mask.contiguous().view(-1, input_n_samples) # M x 3000
        ### filter empty input
        ### for example, when a1(15s) and a2(45s) in one batch
        ### the audios will be padded into 60s and then segmented into 30s like [a11, a12, a21, a22]
        ### we will skip the computation for a12
        select_index = length_to_attention_mask(segments).to(torch.bool).view(-1)
        input_features = input_features[select_index]
        attention_mask = attention_mask[select_index]
        input_features = input_features.transpose(1,2) # M x 3000 x C -> M x C x 3000

        output = super().forward(
            input_features,
            None,  ## qwen2_audio donot support attention_mask
            head_mask,
            output_attentions,
            output_hidden_states,
            return_dict
        )

        last_hidden_state = output.last_hidden_state # M x 1500 x C
        input_lengths = attention_mask.sum(-1) # M
        output_n_samples = last_hidden_state.size(1)
        ### concate the faetures of the same example
        output_last_hidden_state = last_hidden_state.new_zeros(
            bz, output_n_samples * segments.max().item(), last_hidden_state.size(2)
        )
        output_lengths = input_lengths.new_zeros(bz)
        idx = 0
        for i, l in enumerate(segments):
            output_last_hidden_state[i,:output_n_samples*l,:] = \
                last_hidden_state[idx:idx+l].contiguous().view(-1, last_hidden_state.size(2))
            output_lengths[i] = self._get_feat_extract_output_lengths(input_lengths[idx:idx+l]).sum()
            idx += l

        max_length = output_lengths.max()
        output_last_hidden_state = output_last_hidden_state[:,:max_length,:]

        return AudioEncoderOutput(
            last_hidden_state=output_last_hidden_state,
            hidden_states=None,
            attentions=None,
            output_lengths=output_lengths
        )

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        return super()._get_feat_extract_output_lengths(input_lengths)[1] # after avg pooler


class WavLMEncoder(torch.nn.Module):
    """
    WavLM encoder for raw waveform processing.
    Accepts raw audio input (not mel-spectrograms) and outputs embeddings compatible with OmniSpeech.
    Includes temporal adapter for 2x downsampling to match Whisper's temporal resolution.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Load WavLM base model
        if isinstance(config, dict):
            wavlm_config = Wav2Vec2Config(**config)
        else:
            wavlm_config = config

        self.wavlm = Wav2Vec2Model(wavlm_config)

        # Temporal adapter: 2x downsampling to match Whisper's output resolution
        # Whisper downsamples 30s audio @ 16kHz → 1500 frames (2x stride)
        # WavLM outputs at 50Hz (320 samples/frame @ 16kHz), we need 2x downsampling
        self.temporal_adapter = torch.nn.Conv1d(
            in_channels=wavlm_config.hidden_size,
            out_channels=wavlm_config.hidden_size,
            kernel_size=3,
            stride=2,
            padding=1
        )

        # Dimension projection: WavLM (768) → OmniSpeech adapter expects (1280)
        # This matches Whisper/Qwen2Audio's d_model dimension
        self.dimension_projection = torch.nn.Linear(
            wavlm_config.hidden_size,  # 768
            1280  # Match Whisper's d_model for OmniSpeech compatibility
        )

        self.max_source_positions = wavlm_config.max_source_positions if hasattr(wavlm_config, 'max_source_positions') else 1500

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        """Load pretrained WavLM model from HuggingFace"""
        config = Wav2Vec2Config.from_pretrained(model_name_or_path)
        model = cls(config)

        # Load WavLM weights
        pretrained = Wav2Vec2Model.from_pretrained(model_name_or_path)
        model.wavlm = pretrained

        return model

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Compute output lengths after WavLM feature extraction and temporal adapter.
        WavLM uses CNN layers with stride, then we apply 2x downsampling.
        """
        # WavLM CNN feature extractor (7 layers with stride 2, except last with stride 1)
        # Total downsampling: 2^6 = 64x (approx 320 samples per frame @ 16kHz)
        for i in range(6):
            input_lengths = (input_lengths - 1) // 2 + 1

        # Temporal adapter: additional 2x downsampling
        input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths

    def forward(
        self,
        input_values=None,
        input_features=None,  # Alias for compatibility with OmniSpeech (uses Whisper naming)
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        Args:
            input_values: Raw audio waveform tensor (B, T) - NOT mel-spectrograms
            input_features: Alias for input_values (for OmniSpeech compatibility)
            attention_mask: Attention mask (B, T)

        Returns:
            AudioEncoderOutput with last_hidden_state (B, T', 1280) and output_lengths
        """
        # Handle both parameter names for compatibility
        if input_values is None and input_features is None:
            raise ValueError("Either input_values or input_features must be provided")
        if input_values is None:
            input_values = input_features  # OmniSpeech passes input_features

        bz, seq_len = input_values.shape

        # Handle long audio by segmenting (similar to Whisper encoder)
        input_n_samples = self.max_source_positions * 320  # ~30s @ 16kHz (WavLM uses 320 samples/frame)

        if attention_mask is not None:
            input_lengths = attention_mask.sum(-1)
        else:
            input_lengths = torch.full((bz,), seq_len, dtype=torch.long, device=input_values.device)

        segments = torch.ceil(input_lengths / input_n_samples).to(dtype=torch.long)

        # Segment audio if needed
        if seq_len > input_n_samples:
            # Pad to multiple of segment length
            padded_len = segments.max().item() * input_n_samples
            if seq_len < padded_len:
                padding = torch.zeros(bz, padded_len - seq_len, device=input_values.device)
                input_values = torch.cat([input_values, padding], dim=1)
                if attention_mask is not None:
                    mask_padding = torch.zeros(bz, padded_len - seq_len, device=attention_mask.device)
                    attention_mask = torch.cat([attention_mask, mask_padding], dim=1)

            # Reshape into segments
            input_values = input_values.contiguous().view(-1, input_n_samples)
            if attention_mask is not None:
                attention_mask = attention_mask.contiguous().view(-1, input_n_samples)

            # Filter empty segments
            from .utils import length_to_attention_mask
            select_index = length_to_attention_mask(segments).to(torch.bool).view(-1)
            input_values = input_values[select_index]
            if attention_mask is not None:
                attention_mask = attention_mask[select_index]

        # Pass through WavLM
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state  # (M, T_wavlm, 768)

        # Apply temporal adapter for 2x downsampling
        # Conv1d expects (B, C, T) format
        last_hidden_state = last_hidden_state.transpose(1, 2)  # (M, 768, T_wavlm)
        last_hidden_state = self.temporal_adapter(last_hidden_state)  # (M, 768, T'/2)
        last_hidden_state = last_hidden_state.transpose(1, 2)  # (M, T'/2, 768)

        # Apply dimension projection: 768 → 1280 for OmniSpeech compatibility
        last_hidden_state = self.dimension_projection(last_hidden_state)  # (M, T'/2, 1280)

        if seq_len > input_n_samples:
            # Concatenate segments back together
            output_n_samples = last_hidden_state.size(1)
            output_last_hidden_state = last_hidden_state.new_zeros(
                bz, output_n_samples * segments.max().item(), last_hidden_state.size(2)
            )

            if attention_mask is not None:
                segment_lengths = attention_mask.sum(-1)
            else:
                segment_lengths = torch.full((last_hidden_state.size(0),), input_n_samples, dtype=torch.long, device=input_values.device)

            output_lengths = input_lengths.new_zeros(bz)
            idx = 0
            for i, l in enumerate(segments):
                output_last_hidden_state[i, :output_n_samples*l, :] = \
                    last_hidden_state[idx:idx+l].contiguous().view(-1, last_hidden_state.size(2))
                output_lengths[i] = self._get_feat_extract_output_lengths(segment_lengths[idx:idx+l]).sum()
                idx += l

            max_length = output_lengths.max()
            output_last_hidden_state = output_last_hidden_state[:, :max_length, :]
        else:
            output_last_hidden_state = last_hidden_state
            if attention_mask is not None:
                output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            else:
                output_lengths = self._get_feat_extract_output_lengths(input_lengths)

        return AudioEncoderOutput(
            last_hidden_state=output_last_hidden_state,
            hidden_states=None,
            attentions=None,
            output_lengths=output_lengths
        )


AUDIO_ENCODER_MAPPING = {
    "whisper": WhisperEncoder,
    "qwen2_audio_encoder": Qwen2AudioEncoder,
    "qwen2_audio": Qwen2AudioEncoder,
    "wavlm": WavLMEncoder
}