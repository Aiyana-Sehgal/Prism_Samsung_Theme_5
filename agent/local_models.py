#!/usr/bin/env python3
"""
Local, zero-cost STT and TTS for the LiveKit pipeline.

The submission runs entirely on models that execute on the evaluating machine:
no paid APIs, no network calls to any provider at evaluation time.

  STT  faster-whisper (CTranslate2). GPU when CUDA is usable, CPU otherwise.
  TTS  Piper, invoked as a subprocess. CPU only, fast enough to stay ahead of playback.
  LLM  any OpenAI-compatible local server (Ollama by default) -- wired in fdb_agent.py,
       since livekit-plugins-openai already accepts a base_url.

These are plain LiveKit plugin subclasses. The agent's coordination logic
(agent/controller.py) is untouched and model-agnostic: it gates tool calls on
speech timing, not on which model produced the transcript.

Configuration (env vars, all optional):
  FDB_WHISPER_MODEL    faster-whisper size or path (default: small.en)
  FDB_WHISPER_DEVICE   cuda | cpu | auto (default: cpu -- see _load_model for why)
  FDB_WHISPER_COMPUTE  float16 | int8 | auto (default: auto)
  FDB_WHISPER_BEAM     beam size (default: 1; greedy is fastest and near-identical here)
  FDB_PIPER_VOICE      path to a Piper .onnx voice (default: agent/piper/en_US-amy-medium.onnx)
"""

import asyncio
import os
from typing import Optional

import numpy as np
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    stt,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils.audio import AudioBuffer

WHISPER_SAMPLE_RATE = 16000


# ── STT: faster-whisper ──────────────────────────────────────────────
class LocalWhisperSTT(stt.STT):
    """Batch (non-streaming) Whisper. LiveKit's VAD decides the segment boundaries."""

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        language: str = "en",
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model_name = model or os.getenv("FDB_WHISPER_MODEL", "small.en")
        # CPU by default, deliberately. CTranslate2 (faster-whisper's backend) links against
        # CUDA 12, while the pinned torch ships CUDA 13, so a GPU run needs the extra
        # nvidia-*-cu12 wheels (see README). CPU int8 is fast enough for this benchmark and
        # leaves all VRAM to the LLM, which is the component that actually needs it.
        self._device = device or os.getenv("FDB_WHISPER_DEVICE", "cpu")
        self._compute_type = compute_type or os.getenv("FDB_WHISPER_COMPUTE", "auto")
        self._beam_size = int(os.getenv("FDB_WHISPER_BEAM", "1"))
        self._language = language
        self._model = None
        self._load_lock = asyncio.Lock()
        self._cpu_fallback_done = False

    @property
    def model(self) -> str:
        return self._model_name

    def _resolve_device(self) -> tuple[str, str]:
        device, compute = self._device, self._compute_type
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return device, compute

    @staticmethod
    def _preload_cuda12_libs() -> None:
        """Best-effort: make CUDA 12 cuBLAS/cuDNN loadable by CTranslate2.

        CTranslate2 resolves these through the dynamic loader at first encode(), so the
        nvidia-*-cu12 wheels have to be in the process's library scope by then. Loading
        them RTLD_GLOBAL here achieves that without requiring LD_LIBRARY_PATH to be set
        before launch. A no-op when the wheels are not installed.
        """
        import ctypes
        import glob

        for mod_name in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
            try:
                mod = __import__(mod_name, fromlist=["__file__"])
            except ImportError:
                continue
            for so in sorted(glob.glob(os.path.join(os.path.dirname(mod.__file__), "*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    def _load_model(self, force_cpu: bool = False):
        from faster_whisper import WhisperModel

        device, compute = ("cpu", "int8") if force_cpu else self._resolve_device()
        if device == "cuda":
            self._preload_cuda12_libs()
        try:
            return WhisperModel(self._model_name, device=device, compute_type=compute)
        except Exception as exc:
            if device == "cpu":
                raise
            print(
                f"[local_models] CUDA Whisper unavailable ({exc}); falling back to CPU int8",
                flush=True,
            )
            return WhisperModel(self._model_name, device="cpu", compute_type="int8")

    def _to_mono_16k(self, buffer: AudioBuffer) -> np.ndarray:
        frame = rtc.combine_audio_frames(buffer)
        samples = np.frombuffer(bytes(frame.data), dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            samples = samples.reshape(-1, frame.num_channels).mean(axis=1)
        if frame.sample_rate != WHISPER_SAMPLE_RATE and samples.size:
            target_len = int(round(samples.size * WHISPER_SAMPLE_RATE / frame.sample_rate))
            samples = np.interp(
                np.linspace(0.0, samples.size - 1, target_len),
                np.arange(samples.size),
                samples,
            ).astype(np.float32)
        return samples

    def _run_model(self, samples: np.ndarray, language: Optional[str]) -> str:
        segments, _info = self._model.transcribe(
            samples,
            language=language or None,
            beam_size=self._beam_size,
            vad_filter=False,
            # Each benchmark scenario is independent: never let one utterance's text
            # condition the next. This also satisfies the no-caching-across-scenarios rule.
            condition_on_previous_text=False,
        )
        # transcribe() is lazy: consuming the generator is what actually runs the model,
        # so CUDA loader errors surface here rather than at construction.
        return " ".join(seg.text.strip() for seg in segments).strip()

    def _transcribe(self, samples: np.ndarray, language: Optional[str]) -> str:
        try:
            return self._run_model(samples, language)
        except RuntimeError as exc:
            if self._cpu_fallback_done:
                raise
            # Typically "Library libcublas.so.12 is not found" when the CUDA 12 wheels
            # CTranslate2 needs are absent. Rebuild on CPU once and carry on, so a long
            # run degrades in speed instead of dying.
            print(
                f"[local_models] GPU transcription failed ({exc}); switching to CPU int8 "
                "for the rest of this session",
                flush=True,
            )
            self._cpu_fallback_done = True
            self._model = self._load_model(force_cpu=True)
            return self._run_model(samples, language)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        async with self._load_lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)

        lang = language if isinstance(language, str) else self._language
        samples = self._to_mono_16k(buffer)
        text = "" if samples.size == 0 else await asyncio.to_thread(
            self._transcribe, samples, lang
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=utils.shortuuid(),
            alternatives=[stt.SpeechData(language=lang or "en", text=text)],
        )

    def prewarm(self) -> None:
        if self._model is None:
            self._model = self._load_model()


# ── TTS: Piper ───────────────────────────────────────────────────────
def _default_voice_path() -> str:
    return os.getenv(
        "FDB_PIPER_VOICE",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "piper", "en_US-amy-medium.onnx"
        ),
    )


class PiperTTS(tts.TTS):
    """Non-streaming Piper TTS via its Python API (onnxruntime, CPU, no system binary)."""

    def __init__(self, *, voice_path: Optional[str] = None) -> None:
        self._voice_path = voice_path or _default_voice_path()
        if not os.path.exists(self._voice_path):
            raise APIConnectionError(
                f"Piper voice not found: {self._voice_path}. "
                "reproduce.sh downloads it; set FDB_PIPER_VOICE to override."
            )
        from piper import PiperVoice

        # Loaded once per session: ~60 MB of ONNX, and we need the true sample rate here.
        self._voice = PiperVoice.load(self._voice_path)
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=int(self._voice.config.sample_rate),
            num_channels=1,
        )

    @property
    def model(self) -> str:
        return os.path.basename(self._voice_path)

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _PiperChunkedStream(tts=self, input_text=text, conn_options=conn_options)


class _PiperChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: PiperTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._piper: PiperTTS = tts

    def _synthesize_blocking(self) -> bytes:
        chunks = [
            chunk.audio_int16_bytes for chunk in self._piper._voice.synthesize(self.input_text)
        ]
        return b"".join(chunks)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._piper.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )
        # ONNX inference is blocking and CPU-bound; keep it off the event loop so the
        # controller keeps receiving speech events while the reply is synthesised.
        audio = await asyncio.to_thread(self._synthesize_blocking)
        if audio:
            output_emitter.push(audio)
        output_emitter.flush()
