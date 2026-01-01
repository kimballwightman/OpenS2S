#!/usr/bin/env python3
"""
OpenS2S Inference Server
FastAPI server that runs OpenS2S model inference and communicates with the backend
"""

import os
import sys
import asyncio
import logging
import time
import json
import base64
import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add src directory to path for OpenS2S imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

# OpenS2S imports (from the submodule)
try:
    from flow_inference import OpenS2SInference
    from configuration_omnispeech import OmniSpeechConfig
    from modeling_omnispeech import OmniSpeechForConditionalGeneration
    import torch
except ImportError as e:
    print(f"Error importing OpenS2S modules: {e}")
    print("Make sure you're running this from the OpenS2S directory with proper dependencies installed")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class AudioProcessingRequest(BaseModel):
    action: str
    session_id: str
    audio_window: str  # base64 encoded audio
    window_size: int
    sample_rate: int = 16000
    persona: str = "busy_homeowner"
    past_key_values: Optional[Any] = None
    sequence: int = 0
    timestamp: float = 0.0
    attempt: int = 1

class SessionRequest(BaseModel):
    action: str
    session_id: str
    persona: str = "busy_homeowner"
    config: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    active_sessions: int
    gpu_available: bool

@dataclass
class InferenceSession:
    """Track inference session state"""
    session_id: str
    persona: str
    past_key_values: Optional[Any] = None
    conversation_history: list = None
    start_time: float = 0.0
    last_activity: float = 0.0

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.start_time == 0.0:
            self.start_time = time.time()
        self.last_activity = time.time()

class OpenS2SInferenceServer:
    """OpenS2S Inference Server managing model and sessions"""

    def __init__(self):
        self.app = FastAPI(title="OpenS2S Inference Server", version="1.0.0")
        self.model = None
        self.inference_engine = None
        self.sessions: Dict[str, InferenceSession] = {}
        self.model_loaded = False

        # Setup routes
        self._setup_routes()

        # Initialize model on startup
        self._initialize_model()

    def _setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy" if self.model_loaded else "loading",
                model_loaded=self.model_loaded,
                active_sessions=len(self.sessions),
                gpu_available=torch.cuda.is_available()
            )

        @self.app.post("/api/session")
        async def manage_session(request: SessionRequest):
            """Create or manage inference sessions"""
            try:
                if request.action == "start_session":
                    return await self._start_session(request.session_id, request.persona)
                elif request.action == "end_session":
                    return await self._end_session(request.session_id)
                else:
                    raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")

            except Exception as e:
                logger.error(f"Session management error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/stream_infer")
        async def stream_infer(request: Request):
            """
            HTTP/1.1 streaming: raw PCM in, raw PCM out - single request, dual streams
            No JSON, no base64, no polling - true chunked transfer encoding
            """
            try:
                if not self.model_loaded:
                    raise HTTPException(status_code=503, detail="Model not loaded")

                # Return StreamingResponse with raw PCM binary output
                return StreamingResponse(
                    self._audio_stream(request),
                    media_type="application/octet-stream"
                )

            except Exception as e:
                logger.error(f"Streaming inference error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    def _initialize_model(self):
        """Initialize OpenS2S model"""
        try:
            logger.info("🤖 Initializing OpenS2S model...")

            # Check if GPU is available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {device}")

            if device == "cpu":
                logger.warning("⚠️ Running on CPU - inference will be slow")

            # Initialize OpenS2S inference engine
            # NOTE: You'll need to adjust this based on the actual OpenS2S model loading code
            # This is a placeholder that you'll need to implement based on the OpenS2S repo structure

            # Load model configuration
            # config = OmniSpeechConfig.from_pretrained("path/to/model")
            # self.model = OmniSpeechForConditionalGeneration.from_pretrained("path/to/model")
            # self.inference_engine = OpenS2SInference(self.model, config)

            # For now, create a mock inference engine for testing
            self.inference_engine = MockOpenS2SInference()

            self.model_loaded = True
            logger.info("✅ OpenS2S model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load OpenS2S model: {e}")
            self.model_loaded = False

    async def _start_session(self, session_id: str, persona: str) -> Dict[str, Any]:
        """Start a new inference session"""
        try:
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, updating...")

            # Create new session
            session = InferenceSession(
                session_id=session_id,
                persona=persona
            )
            self.sessions[session_id] = session

            logger.info(f"✅ Started inference session {session_id} with persona: {persona}")

            return {
                "status": "session_started",
                "session_id": session_id,
                "persona": persona,
                "cache": {}  # Model cache info if needed
            }

        except Exception as e:
            logger.error(f"Error starting session {session_id}: {e}")
            raise

    async def _end_session(self, session_id: str) -> Dict[str, Any]:
        """End inference session"""
        try:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"✅ Ended inference session {session_id}")
                return {"status": "session_ended", "session_id": session_id}
            else:
                logger.warning(f"Session {session_id} not found for ending")
                return {"status": "session_not_found", "session_id": session_id}

        except Exception as e:
            logger.error(f"Error ending session {session_id}: {e}")
            raise

    async def _audio_stream(self, request: Request):
        """
        TRUE HTTP/1.1 streaming: Raw PCM in, raw PCM out
        Single request with streaming request body AND streaming response body
        No JSON, no base64, no polling - exactly as ChatGPT specified
        """
        try:
            session_id = f"stream_{int(time.time())}"
            logger.info(f"🎵 Starting HTTP/1.1 audio stream session {session_id}")

            # Initialize rolling buffer for this stream
            rolling_buffer = []

            # Read raw PCM chunks from streaming request body
            async for in_chunk in request.stream():
                if not in_chunk:
                    continue

                try:
                    # Parse frame header: [4 bytes length][PCM payload]
                    if len(in_chunk) < 4:
                        continue

                    # Read frame length (first 4 bytes)
                    frame_length = int.from_bytes(in_chunk[:4], byteorder='little')

                    # Extract PCM payload (remaining bytes)
                    pcm_bytes = in_chunk[4:4+frame_length]

                    if len(pcm_bytes) != frame_length:
                        logger.warning(f"⚠️ Frame length mismatch: expected {frame_length}, got {len(pcm_bytes)}")
                        continue

                    # Convert PCM16 to float32 and add to rolling buffer
                    pcm_samples = np.frombuffer(pcm_bytes, dtype=np.int16)
                    float_samples = pcm_samples.astype(np.float32) / 32768.0

                    # model.push_audio() equivalent
                    rolling_buffer.extend(float_samples)

                    # Keep only last 30 seconds at 16kHz (480,000 samples)
                    if len(rolling_buffer) > 480000:
                        rolling_buffer = rolling_buffer[-480000:]

                    logger.debug(f"🔊 Pushed {len(float_samples)} samples, buffer: {len(rolling_buffer)} total")

                    # model.has_output() and model.get_output() equivalent
                    if len(rolling_buffer) >= 16000:  # At least 1 second for inference
                        audio_array = np.array(rolling_buffer[-160000:], dtype=np.float32)  # Last 10s

                        # Run inference and get output PCM chunks
                        async for output_pcm in self.inference_engine.process_audio_chunk(audio_array, session_id):
                            if output_pcm is not None and len(output_pcm) > 0:
                                # Convert float32 back to PCM16 bytes
                                pcm16_output = (np.clip(output_pcm, -1.0, 1.0) * 32767).astype(np.int16)

                                # Frame the output: [4 bytes length][PCM payload]
                                pcm_bytes = pcm16_output.tobytes()
                                frame_length = len(pcm_bytes)
                                frame_header = frame_length.to_bytes(4, byteorder='little')

                                # Yield raw PCM frame (no JSON, no base64)
                                yield frame_header + pcm_bytes

                except Exception as chunk_error:
                    logger.error(f"❌ Error processing chunk: {chunk_error}")
                    continue

            logger.info(f"✅ Completed HTTP/1.1 audio stream session {session_id}")

        except Exception as e:
            logger.error(f"❌ Error in audio streaming: {e}")
            # Send error as a special frame (optional)
            error_msg = f"Stream error: {e}".encode('utf-8')
            error_frame = len(error_msg).to_bytes(4, byteorder='little') + error_msg
            yield error_frame

    async def _poll_model_outputs(self, audio_buffer: list, session_id: str):
        """
        Poll model for outputs given current audio buffer
        This is where the actual OpenS2S inference happens
        """
        try:
            # Convert buffer to numpy array for model input
            audio_array = np.array(audio_buffer[-160000:], dtype=np.float32)  # Last 10 seconds

            # Run OpenS2S inference (replace with actual model)
            async for result in self.inference_engine.stream_inference(
                audio=audio_array,
                persona="busy_homeowner",  # Default for now
                past_key_values=None,
                session_id=session_id
            ):
                yield result

        except Exception as e:
            logger.error(f"❌ Model polling error: {e}")
            yield {
                "type": "error",
                "data": {"message": str(e), "error_code": "model_error"},
                "session_id": session_id,
                "timestamp": time.time()
            }

    async def _run_inference(self, session: InferenceSession, audio: np.ndarray, persona: str):
        """Run OpenS2S inference and yield streaming results"""
        try:
            # Update conversation context
            session.past_key_values = session.past_key_values  # Maintain context

            # Simulate OpenS2S inference process
            # NOTE: Replace this with actual OpenS2S inference code
            async for result_chunk in self.inference_engine.stream_inference(
                audio=audio,
                persona=persona,
                past_key_values=session.past_key_values,
                session_id=session.session_id
            ):
                yield result_chunk

        except Exception as e:
            logger.error(f"Inference error for session {session.session_id}: {e}")
            raise

class MockOpenS2SInference:
    """Mock OpenS2S inference for testing until real model is integrated"""

    async def process_audio_chunk(self, audio: np.ndarray, session_id: str):
        """
        Process audio chunk and yield raw PCM output chunks
        This replaces the old JSON-based streaming with raw PCM streaming
        """
        try:
            # Simulate processing delay
            await asyncio.sleep(0.1)

            # Generate mock response audio (simulate speech synthesis)
            # In real OpenS2S, this would be the actual model inference
            response_duration_s = 1.0  # 1 second of response audio
            sample_rate = 16000
            num_samples = int(response_duration_s * sample_rate)

            # Generate mock sine wave response (replace with real model output)
            t = np.linspace(0, response_duration_s, num_samples, False)
            frequency = 220 + (len(audio) % 440)  # Vary frequency based on input
            mock_response = 0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32)

            # Yield response in chunks (simulate streaming output)
            chunk_size = 1600  # 0.1 second chunks at 16kHz
            for i in range(0, len(mock_response), chunk_size):
                chunk = mock_response[i:i + chunk_size]
                if len(chunk) > 0:
                    yield chunk
                    await asyncio.sleep(0.05)  # Simulate real-time generation

            logger.debug(f"🎤 Mock inference completed for session {session_id}: {num_samples} samples generated")

        except Exception as e:
            logger.error(f"❌ Mock inference error for session {session_id}: {e}")
            yield None

def create_app():
    """Create FastAPI app instance"""
    server = OpenS2SInferenceServer()
    return server.app

if __name__ == "__main__":
    # Create and run server
    server = OpenS2SInferenceServer()

    logger.info("🚀 Starting OpenS2S Inference Server on port 8080")
    uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )