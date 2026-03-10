"""
A model worker executes the model.
"""
import os
import argparse
import asyncio
import json
import time
import threading
import uuid
import sys
from copy import deepcopy
import base64
import tempfile
import shutil
import logging
from io import BytesIO
import torchaudio

from fastapi import FastAPI, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
import requests
import torch
import uvicorn
import numpy as np
from functools import partial

from transformers import AutoTokenizer, AutoConfig
from transformers import TextIteratorStreamer
from transformers import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from threading import Thread
from queue import Queue

from flow_inference import AudioDecoder
from src.constants import WORKER_HEART_BEAT_INTERVAL
from src.constants import DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN, DEFAULT_TTS_START_TOKEN
from src.constants import DEFAULT_AUDIO_TOKEN, AUDIO_TOKEN_INDEX
from src.modeling_omnispeech import OmniSpeechModel
from src.utils import get_waveform
# Note: WhisperFeatureExtractor removed - WavLM processes raw audio directly

sys.path.append("cosyvoice")
sys.path.append("third_party/Matcha-TTS")

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
global_counter = 0
model_semaphore = None

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("worker")

# WavLM processor for streaming mode (import after logger setup)
try:
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
    WAVLM_AVAILABLE = True
except ImportError:
    logger.warning("WavLM not available. Install transformers[torch] for streaming mode support.")
    WAVLM_AVAILABLE = False

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"

class TokenStreamer(BaseStreamer):
    def __init__(self, skip_prompt: bool = False, timeout=None):
        self.skip_prompt = skip_prompt

        # variables used in the streaming process
        self.token_queue = Queue()
        self.stop_signal = None
        self.next_tokens_are_prompt = True
        self.timeout = timeout

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        for token in value.tolist():
            self.token_queue.put(token)

    def end(self):
        self.token_queue.put(self.stop_signal)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.token_queue.get(timeout=self.timeout)
        if value == self.stop_signal:
            raise StopIteration()
        else:
            return value


def heart_beat_worker(controller):
    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"

def load_pretrained_model(audio_encoder_path, llm_path, tts_path, adapters_path):
    """
    Load OmniSpeech model from 4 separate HuggingFace repos with WavLM audio encoder.

    This function manually loads each sub-model from separate repos and composes them
    into a single OmniSpeechModel instance.

    Args:
        audio_encoder_path: Path to WavLM audio encoder repo (local or HF)
        llm_path: Path to Qwen3 7B LLM repo
        tts_path: Path to Qwen3 2B TTS repo
        adapters_path: Path to adapter parameters repo

    Returns:
        tokenizer, tts_tokenizer, model, generation_config, tts_generation_config

    Memory Profile:
        - With GPTQ-quantized LLM (8-bit): ~12-14GB
        - Full precision (bf16): ~20-24GB
    """
    logger.info("📦 Loading OmniSpeech model from separate repos...")
    logger.info(f"   Audio Encoder: {audio_encoder_path}")
    logger.info(f"   LLM: {llm_path}")
    logger.info(f"   TTS: {tts_path}")
    logger.info(f"   Adapters: {adapters_path}")

    # Import required modules
    import sys
    sys.path.insert(0, './src')
    from configuration_omnispeech import OmniSpeechConfig
    from modeling_omnispeech import OmniSpeechModel
    from modeling_tts_lm import TTS_LM_MAPPING

    # Step 1: Load configs from separate repos
    logger.info("\n📋 Step 1: Loading configs...")
    config = OmniSpeechConfig.from_separate_repos(
        audio_encoder_repo=audio_encoder_path,
        llm_repo=llm_path,
        tts_repo=tts_path,
        adapter_repo=adapters_path
    )
    logger.info("   ✅ Configs loaded (WavLM audio encoder forced)")

    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
    tts_tokenizer = AutoTokenizer.from_pretrained(tts_path, local_files_only=True)

    # Load generation configs
    generation_config = GenerationConfig.from_pretrained(llm_path, local_files_only=True)
    tts_generation_config = GenerationConfig.from_pretrained(tts_path, local_files_only=True)

    # Step 2: Create OmniSpeechModel instance (structure only, random weights)
    logger.info("\n🏗️  Step 2: Creating OmniSpeechModel instance...")
    model = OmniSpeechModel(config)
    logger.info("   ✅ Model structure created")

    # Step 3: Load sub-model weights individually
    logger.info("\n💾 Step 3: Loading sub-model weights...")

    # Load Audio Encoder (WavLMEncoder weights)
    logger.info("   🎤 Loading WavLM audio encoder...")
    audio_encoder_state = torch.load(
        os.path.join(audio_encoder_path, "pytorch_model.bin"),
        map_location="cpu"
    )
    model.audio_encoder_model.load_state_dict(audio_encoder_state)
    model.audio_encoder_model.to(torch.bfloat16)
    logger.info("   ✅ Audio encoder loaded (~768MB)")

    # Check for GPTQ-quantized LLM first
    gptq_llm_path = "/models/SalesS2S-llm-gptq"
    use_gptq_llm = os.path.exists(gptq_llm_path) and os.listdir(gptq_llm_path)

    # Load LLM
    logger.info("   🧠 Loading LLM...")
    if use_gptq_llm:
        logger.info("   🔄 Using GPTQ-quantized LLM...")
        try:
            from auto_gptq import AutoGPTQForCausalLM
            llm = AutoGPTQForCausalLM.from_quantized(
                gptq_llm_path,
                device="cuda:0",
                use_safetensors=True
            )
            model.llm_model = llm
            logger.info("   ✅ GPTQ LLM loaded (8-bit, ~8GB)")
        except Exception as e:
            logger.warning(f"   ⚠️  GPTQ load failed: {e}, using full precision")
            use_gptq_llm = False

    if not use_gptq_llm:
        # Load full precision LLM
        llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True
        )
        model.llm_model = llm
        logger.info("   ✅ LLM loaded (bf16, ~14GB)")

    # Load TTS LM
    logger.info("   🎵 Loading TTS LM...")
    tts_model = TTS_LM_MAPPING[config.tts_lm_config.model_type](config.tts_lm_config)

    # Load pretrained weights into TTS model
    tts_state = torch.load(
        os.path.join(tts_path, "pytorch_model.bin"),
        map_location="cpu"
    )
    tts_model.load_state_dict(tts_state, strict=False)
    tts_model.to(torch.bfloat16)

    model.tts_lm_model = tts_model
    logger.info("   ✅ TTS LM loaded (~4GB)")

    # Step 4: Load adapter weights
    logger.info("\n🔌 Step 4: Loading adapter parameters...")
    adapter_weights_path = os.path.join(adapters_path, "adapter_weights.pt")
    adapter_state = torch.load(adapter_weights_path, map_location="cpu")

    # Load adapter parameters into model
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    logger.info(f"   ✅ Loaded {len(adapter_state)} adapter parameters (~146MB)")

    if missing:
        logger.info(f"   ℹ️  Missing keys (expected): {len(missing)}")
    if unexpected:
        logger.warning(f"   ⚠️  Unexpected keys: {unexpected}")

    # Summary
    logger.info("\n✅ Model loading complete!")
    logger.info(f"   Audio Encoder: WavLMEncoder (768 hidden, 12 layers)")
    logger.info(f"   LLM: {'GPTQ-quantized (8-bit)' if use_gptq_llm else 'Full precision (bf16)'}")
    logger.info(f"   TTS: Qwen3 2B (bf16)")
    logger.info(f"   Adapters: Loaded from {adapters_path}")
    logger.info(f"   Ready for inference")

    return tokenizer, tts_tokenizer, model, generation_config, tts_generation_config

def load_flow_model(flow_path):
    flow_config = os.path.join(args.flow_path, "config.yaml")
    flow_checkpoint = os.path.join(args.flow_path, 'flow.pt')
    hift_checkpoint = os.path.join(args.flow_path, 'hift.pt')
    audio_decoder = AudioDecoder(flow_config, flow_checkpoint, hift_checkpoint, device="cuda")
    return audio_decoder


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register, audio_encoder_path,
                 llm_path, tts_path, adapters_path,
                 flow_path, model_name):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        self.model_name = model_name
        self.audio_processor_type = "wavlm"  # Hardcoded - WavLM is the only encoder now

        # Load main models from separate repos (WavLM + LLM + TTS + Adapters)
        self.tokenizer, self.tts_tokenizer, self.model, self.generation_config,\
            self.tts_generation_config = load_pretrained_model(
                audio_encoder_path, llm_path, tts_path, adapters_path
            )

        # Initialize audio processors (WavLM only) - loads microsoft/wavlm-base-plus for streaming
        self._init_audio_processors()

        self.audio_decoder = load_flow_model(flow_path)
        self.system_prompt = "You are a helpful assistant."
        self.units_bias = self.tts_tokenizer.encode("<|audio_0|>")[0]

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,), daemon=True)
            self.heart_beat_thread.start()

    def _init_audio_processors(self):
        """Initialize audio processors - WavLM only"""
        logger.info("Initializing audio processor: wavlm (hardcoded)")

        if not WAVLM_AVAILABLE:
            logger.error("❌ WavLM not available - transformers[torch] required!")
            logger.error("   Install with: pip install transformers[torch]")
            raise ImportError("WavLM (Wav2Vec2) modules not available")

        logger.info("Loading WavLM model for streaming mode...")
        self.wavlm_model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
        self.wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.audio_extractor = self.wavlm_feature_extractor
        logger.info("✅ WavLM model loaded successfully")

    def process_audio_streaming(self, audio_samples, sample_rate=16000):
        """Process audio using WavLM (only encoder supported)"""
        # Convert to format expected by WavLM
        if isinstance(audio_samples, np.ndarray):
            if audio_samples.dtype != np.float32:
                audio_samples = audio_samples.astype(np.float32)

        # WavLM processing
        inputs = self.wavlm_feature_extractor(
            audio_samples,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )

        # Get WavLM features
        with torch.no_grad():
            outputs = self.wavlm_model(**inputs)

        # Return in format compatible with existing pipeline
        return {
            "input_features": outputs.last_hidden_state,
            "attention_mask": inputs.get("attention_mask", None)
        }

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    def get_input_params(self, messages):
        new_messages = []
        audios = []
        if self.system_prompt:
            new_messages.append({"role": "system", "content": self.system_prompt})
        for turn in messages:
            role = turn["role"]
            content = turn["content"]
            if isinstance(content, str):
                new_content = content
            elif isinstance(content, list):
                new_content = ""
                for item in content:
                    if item.get("audio", ""):
                        audio_binary = base64.b64decode(item["audio"])
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                            temp_file.write(audio_binary)
                            temp_file_path = temp_file.name
                            waveform = get_waveform(temp_file_path)
                            audios.append(waveform)
                        new_content += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                    elif item.get("text", ""):
                        new_content += item["text"]
            elif isinstance(content, dict):
                new_content = ""
                if content.get("audio", ""):
                    audio_binary = base64.b64decode(content["audio"])
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
                        temp_file.write(audio_binary)
                        temp_file_path = temp_file.name
                        waveform = get_waveform(temp_file_path)
                        audios.append(waveform)
                    new_content += f"{DEFAULT_AUDIO_START_TOKEN}{DEFAULT_AUDIO_TOKEN}{DEFAULT_AUDIO_END_TOKEN}"
                elif content.get("text", ""):
                    new_content += content["text"]
            else:
                raise NotImplementedError
            new_messages.append({"role": f"{role}", "content": f"{new_content}"})

        prompt = self.tokenizer.apply_chat_template(new_messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        prompt += DEFAULT_TTS_START_TOKEN
        segments = prompt.split(f"{DEFAULT_AUDIO_TOKEN}")
        input_ids = []
        for idx, segment in enumerate(segments):
            if idx != 0:
                input_ids += [AUDIO_TOKEN_INDEX]
            input_ids += self.tokenizer.encode(segment)
        input_ids = torch.LongTensor(input_ids).unsqueeze(0)

        if audios:
            # WavLM expects raw audio waveform (not mel-spectrograms)
            speech_inputs = self.audio_extractor(
                audios,  # List of raw waveforms (numpy arrays)
                sampling_rate=16000,
                return_attention_mask=True,
                return_tensors="pt"
            )
            # WavLM feature extractor returns input_values (raw audio)
            speech_values = speech_inputs.input_values
            speech_mask = speech_inputs.attention_mask
        else:
            speech_values, speech_mask = None, None

        return input_ids, speech_values, speech_mask

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model = self.tokenizer, self.model
        generation_config = deepcopy(self.generation_config)
        tts_generation_config = deepcopy(self.tts_generation_config)

        messages = params["messages"]
        input_ids, speech_values, speech_mask = self.get_input_params(messages)
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        if speech_values is not None:
            speech_values = speech_values.to(dtype=torch.bfloat16, device='cuda', non_blocking=True)
            speech_mask = speech_mask.to(device='cuda', non_blocking=True)

        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        do_sample = True if temperature > 0.001 else False

        generation_config.update(
            **{
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
        )
        tts_generation_config.update(
            **{
                "do_sample": True,
                "temperature": 1.0,
                "top_p": 1.0
            }
        )

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True, timeout=15)
        units_streamer = TokenStreamer(skip_prompt=False, timeout=15)

        thread = Thread(target=model.generate, kwargs=dict(
            input_ids=input_ids,
            attention_mask=None,
            speech_values=speech_values,
            speech_mask=speech_mask,
            spk_emb=None,
            units_gen=True,
            streamer=streamer,
            units_streamer=units_streamer,
            generation_config=generation_config,
            tts_generation_config=tts_generation_config,
            use_cache=True,
        ))
        thread.start()

        generated_text = ""
        units = []
        this_uuid = uuid.uuid4()
        prompt_speech_feat = torch.zeros(1, 0, 80).to(device='cuda')
        flow_prompt_speech_token = torch.zeros(1, 0, dtype=torch.int64).to(device='cuda')
        tts_mels = []
        prev_mel = None
        block_size = 24
        iter_text = iter(streamer)
        iter_units = iter(units_streamer)
        active_text = True
        active_units = True
        while active_text or active_units:
            if active_text:
                try:
                    new_text = next(iter_text)
                    generated_text += new_text
                    yield json.dumps({"text": generated_text, "audio": "", "finalize": False, "error_code": 0}).encode() + b"\0"
                except StopIteration:
                    active_text = False
            if active_units:
                try:
                    new_unit = next(units_streamer)
                    units.append(new_unit - self.units_bias)
                    if len(units) >= block_size:
                        tts_token = torch.LongTensor(units).unsqueeze(0).to(device='cuda')
                        if prev_mel is not None:
                            prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                        tts_speech, tts_mel = self.audio_decoder.token2wav(tts_token, uuid=this_uuid,
                            prompt_token=flow_prompt_speech_token.to(device='cuda'),
                            prompt_feat=prompt_speech_feat.to(device='cuda'),
                            finalize=False)
                        prev_mel = tts_mel
                        tts_mels.append(tts_mel)
                        generated_audio = tts_speech.cpu()
                        buffer = BytesIO()
                        torchaudio.save(buffer, generated_audio, 22050, format="wav")
                        audio_binary = buffer.getvalue()
                        base64_string = base64.b64encode(audio_binary).decode("utf-8")
                        flow_prompt_speech_token = torch.cat((flow_prompt_speech_token, tts_token), dim=-1)
                        units = []
                        yield json.dumps({"text": generated_text, "audio": base64_string, "finalize": False, "error_code": 0}).encode() + b"\0"
                except StopIteration:
                    active_units = False
                    if units:
                        tts_token = torch.LongTensor(units).unsqueeze(0).to(device='cuda')
                        if prev_mel is not None:
                            prompt_speech_feat = torch.cat(tts_mels, dim=-1).transpose(1, 2)
                        tts_speech, tts_mel = self.audio_decoder.token2wav(tts_token, uuid=this_uuid,
                            prompt_token=flow_prompt_speech_token.to(device='cuda'),
                            prompt_feat=prompt_speech_feat.to(device='cuda') if prev_mel is not None else None,
                            finalize=True)
                        generated_audio = tts_speech.cpu()
                        buffer = BytesIO()
                        torchaudio.save(buffer, generated_audio, 22050, format="wav")
                        audio_binary = buffer.getvalue()
                        base64_string = base64.b64encode(audio_binary).decode("utf-8")
                        units = []
                        yield json.dumps({"text": generated_text, "audio": base64_string, "finalize": True, "error_code": 0}).encode() + b"\0"
                    else:
                        yield json.dumps({"text": generated_text, "audio": "", "finalize": True, "error_code": 0}).encode() + b"\0"


    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


@app.get("/health")
async def health():
    """Simple health check endpoint for Docker healthcheck and iOS connection testing"""
    return {"status": "healthy", "service": "model_worker"}


# WebSocket session storage for streaming mode
websocket_sessions = {}

@app.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming audio processing with rolling window context.
    Supports manual trigger for response generation.

    Protocol:
    - Audio chunks: binary PCM16 data (16kHz, mono)
    - Control messages: JSON with {"type": "trigger_response", "timestamp": 1234567890}
    - Responses: binary chunks with [4-byte length header] + [PCM16 audio data]
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    logger.info(f"[WEBSOCKET] 🔗 WebSocket connection established for session {session_id}")

    # Initialize session state with rolling window buffer
    websocket_sessions[session_id] = {
        "audio_buffer": RollingWindowAudioBuffer(max_duration_seconds=30),
        "model_cache": {},
        "last_activity": time.time(),
        "is_processing": False,
        "websocket": websocket
    }

    try:
        while True:
            # Receive message from client
            data = await websocket.receive()

            if "bytes" in data:
                # Binary audio chunk
                await handle_audio_chunk(session_id, data["bytes"])
            elif "text" in data:
                # JSON control message
                message = json.loads(data["text"])
                if message.get("type") == "trigger_response":
                    await handle_manual_trigger(session_id)
                else:
                    logger.warning(f"[WEBSOCKET] Unknown message type: {message.get('type')}")

            # Update last activity
            websocket_sessions[session_id]["last_activity"] = time.time()

    except WebSocketDisconnect:
        logger.info(f"[WEBSOCKET] 🔌 Client disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"[WEBSOCKET] ❌ Error in session {session_id}: {e}")
    finally:
        # Cleanup session
        if session_id in websocket_sessions:
            del websocket_sessions[session_id]
            logger.info(f"[WEBSOCKET] 🧹 Cleaned up session {session_id}")


class RollingWindowAudioBuffer:
    """30-second rolling window audio buffer for streaming mode"""

    def __init__(self, max_duration_seconds=30, sample_rate=16000):
        self.max_duration_seconds = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.is_full = False
        self.total_samples_received = 0

    def add_audio_chunk(self, pcm_chunk: bytes):
        """Add PCM16 audio chunk to rolling buffer"""
        # Convert PCM16 bytes to float32 samples
        samples = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        chunk_size = len(samples)

        # Handle buffer wraparound
        if self.write_pos + chunk_size > self.max_samples:
            # Split write across end and beginning of buffer
            end_space = self.max_samples - self.write_pos
            self.buffer[self.write_pos:] = samples[:end_space]
            self.buffer[:chunk_size - end_space] = samples[end_space:]
            self.write_pos = chunk_size - end_space
            self.is_full = True
        else:
            self.buffer[self.write_pos:self.write_pos + chunk_size] = samples
            self.write_pos += chunk_size

        self.total_samples_received += chunk_size

    def get_current_window(self, duration_seconds=None):
        """Get current audio window for processing"""
        if duration_seconds is None:
            duration_seconds = self.max_duration_seconds

        window_samples = int(duration_seconds * self.sample_rate)

        if not self.is_full:
            # Buffer not full yet, return what we have
            return self.buffer[:self.write_pos]
        else:
            # Return rolling window
            if self.write_pos >= window_samples:
                return self.buffer[self.write_pos - window_samples:self.write_pos]
            else:
                # Wrap around case
                return np.concatenate([
                    self.buffer[-(window_samples - self.write_pos):],
                    self.buffer[:self.write_pos]
                ])


async def handle_audio_chunk(session_id: str, audio_chunk: bytes):
    """Handle incoming audio chunk for streaming session"""
    session = websocket_sessions.get(session_id)
    if not session:
        return

    # Add to rolling buffer
    session["audio_buffer"].add_audio_chunk(audio_chunk)

    # Process with WavLM (only audio processor supported)
    # Get recent audio window for incremental WavLM processing
    recent_window = session["audio_buffer"].get_current_window(duration_seconds=3.0)

    # Process with WavLM if we have enough audio
    if len(recent_window) > 0:
        try:
            audio_features = worker.process_audio_streaming(recent_window, sample_rate=16000)
            # Store features in session cache for trigger processing
            session["model_cache"]["recent_features"] = audio_features
            logger.debug(f"[WEBSOCKET] 🔊 WavLM processed {len(recent_window)} samples for session {session_id}")
        except Exception as e:
            logger.error(f"[WEBSOCKET] ❌ WavLM processing error for session {session_id}: {e}")

    logger.debug(f"[WEBSOCKET] 📊 Session {session_id}: Added {len(audio_chunk)} bytes, "
                f"total samples: {session['audio_buffer'].total_samples_received}")


async def handle_manual_trigger(session_id: str):
    """Handle manual trigger for response generation"""
    session = websocket_sessions.get(session_id)
    if not session or session["is_processing"]:
        return

    session["is_processing"] = True
    websocket = session["websocket"]

    try:
        logger.info(f"[WEBSOCKET] 🎯 Manual trigger activated for session {session_id}")

        # Acquire model semaphore for generation
        global model_semaphore
        if model_semaphore is None:
            model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
        await model_semaphore.acquire()

        try:
            # Get current audio context window (30s) as raw PCM float32
            audio_window = session["audio_buffer"].get_current_window()

            # Use cached WavLM features if available, otherwise process now
            if "recent_features" in session["model_cache"]:
                audio_features = session["model_cache"]["recent_features"]
                logger.debug(f"[WEBSOCKET] Using cached WavLM features")
            else:
                logger.debug(f"[WEBSOCKET] Processing audio window with WavLM")
                audio_features = worker.process_audio_streaming(audio_window, sample_rate=16000)

            # TODO: Pass audio_features directly to model inference
            # For now, convert to WAV for compatibility with existing generate_stream
            audio_wav_bytes = convert_float32_to_wav(audio_window, sample_rate=16000)

            # Create messages structure - keep base64 for now until generate_stream supports raw PCM
            # TODO: Refactor generate_stream to accept raw PCM audio features directly
            audio_base64 = base64.b64encode(audio_wav_bytes).decode()
            messages = [
                {
                    "role": "user",
                    "content": audio_base64
                }
            ]

            params = {
                "messages": messages,
                "temperature": 1.0,
                "top_p": 1.0,
                "max_new_tokens": 256
            }

            # Use existing model pipeline for generation
            for response_chunk in worker.generate_stream_gate(params):
                # Parse response chunk (null-terminated JSON)
                if response_chunk.endswith(b"\0"):
                    response_data = json.loads(response_chunk[:-1].decode())

                    # If there's audio in the response, send as binary
                    if "audio" in response_data and response_data["audio"]:
                        audio_data = base64.b64decode(response_data["audio"])
                        # Send binary frame: [4-byte length] + [audio data]
                        length_header = len(audio_data).to_bytes(4, byteorder='little')
                        await websocket.send_bytes(length_header + audio_data)

                    # Check if generation is complete
                    if response_data.get("finalize", False):
                        break

                # Yield control to event loop to keep async context responsive
                await asyncio.sleep(0)

        finally:
            # Clear CUDA cache after generation to free memory for next request
            # This preserves conversation context (stored as text/audio) while
            # freeing GPU memory used for temporary tensors during inference
            torch.cuda.empty_cache()
            model_semaphore.release()

    except Exception as e:
        logger.error(f"[WEBSOCKET] ❌ Error in manual trigger for session {session_id}: {e}")
    finally:
        session["is_processing"] = False


def convert_float32_to_wav(audio_samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 audio samples to WAV format bytes"""
    # Convert float32 to int16
    if len(audio_samples) == 0:
        audio_samples = np.zeros(int(0.1 * sample_rate), dtype=np.float32)  # 100ms silence

    audio_int16 = (audio_samples * 32767).astype(np.int16)

    # Create WAV file in memory
    wav_buffer = BytesIO()
    torchaudio.save(wav_buffer, torch.from_numpy(audio_int16).unsqueeze(0), sample_rate, format="wav")
    wav_buffer.seek(0)
    return wav_buffer.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-name", type=str, default="omnispeech")
    parser.add_argument("--audio-encoder-path", type=str, required=True,
        help="Path to WavLM audio encoder repo")
    parser.add_argument("--llm-path", type=str, required=True,
        help="Path to Qwen3 7B LLM repo")
    parser.add_argument("--tts-path", type=str, required=True,
        help="Path to Qwen3 2B TTS repo")
    parser.add_argument("--adapters-path", type=str, required=True,
        help="Path to adapter parameters repo")
    parser.add_argument("--flow-path", type=str, required=True,
        help="Path to GLM-4 Voice Decoder repo")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.audio_encoder_path,
                         args.llm_path,
                         args.tts_path,
                         args.adapters_path,
                         args.flow_path,
                         args.model_name
                         )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
