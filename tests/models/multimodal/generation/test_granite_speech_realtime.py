# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from dataclasses import asdict

import pytest
import pytest_asyncio

from vllm import LLM, EngineArgs, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

from ....utils import ROCM_ENGINE_KWARGS

MODEL_NAME = "ibm-granite/granite-speech-3.3-2b"
ENGINE_CONFIG = {
    "model": MODEL_NAME,
    "max_model_len": 2048,
    "max_num_seqs": 4,
    "limit_mm_per_prompt": {"audio": 1},
    "enforce_eager": True,
    "enable_lora": True,
    "max_lora_rank": 64,
    "gpu_memory_utilization": 0.9,
    **ROCM_ENGINE_KWARGS,
}


@pytest.fixture
def audio_assets() -> list[AudioAsset]:
    return [AudioAsset("mary_had_lamb")]


@pytest.fixture
def engine():
    engine_args = EngineArgs(**ENGINE_CONFIG)
    llm = LLM(**asdict(engine_args))
    try:
        yield llm
    finally:
        with contextlib.suppress(Exception):
            llm.llm_engine.engine_core.shutdown()
        import torch

        torch.accelerator.empty_cache()


@pytest_asyncio.fixture
async def async_engine():
    engine_args = AsyncEngineArgs(**ENGINE_CONFIG)
    llm = AsyncLLM.from_engine_args(engine_args)
    try:
        yield llm
    finally:
        llm.shutdown()


def test_granite_speech_realtime_forward(audio_assets, engine):
    """Test that the realtime model can do a basic forward pass.

    Uses the same prompt template as the realtime streaming path
    but in a single-shot (non-streaming) call to verify the model
    produces reasonable transcription output.
    """
    from vllm.lora.request import LoRARequest

    lora_request = LoRARequest("audio", 1, MODEL_NAME)

    audio, sr = audio_assets[0].audio_and_sample_rate
    assert sr == 16000

    prompt = (
        "<|start_of_role|>user<|end_of_role|>"
        "<|audio|>can you transcribe the speech into a written format?"
        "<|end_of_text|>\n"
        "<|start_of_role|>assistant<|end_of_role|>"
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    outputs = engine.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"audio": [audio]},
        },
        sampling_params=sampling_params,
        lora_request=lora_request,
    )

    text = outputs[0].outputs[0].text
    # The audio is "Mary had a little lamb" - verify key words appear
    text_lower = text.lower()
    assert "mary" in text_lower, f"Expected 'mary' in output, got: {text!r}"
    assert "lamb" in text_lower, f"Expected 'lamb' in output, got: {text!r}"


@pytest.mark.asyncio
async def test_granite_speech_realtime_streaming(audio_assets, async_engine):
    """Test the realtime streaming generator with segmented audio.

    Feeds audio through the buffer_realtime_audio async generator
    and verifies that the model produces transcription output
    across streaming segments.
    """
    # Lazy imports to avoid CUDA-reinitialization error
    import numpy as np

    from vllm.model_executor.models.granite_speech_realtime import (
        GraniteSpeechRealtimeGeneration,
        RealtimeAudioBuffer,
    )

    audio, sr = audio_assets[0].audio_and_sample_rate
    assert sr == 16000

    # Simulate streaming: split audio into small chunks (~0.5s each)
    chunk_size = sr // 2  # 8000 samples
    audio_chunks = [audio[i : i + chunk_size] for i in range(0, len(audio), chunk_size)]

    # Feed chunks through the buffer to verify segmentation works
    buffer = RealtimeAudioBuffer(sampling_rate=sr, segment_duration_s=5.0)
    segment_count = 0
    for chunk in audio_chunks:
        buffer.write_audio(chunk)
        while buffer.read_audio() is not None:
            segment_count += 1

    remaining = buffer.flush()
    if remaining is not None and len(remaining) > 0:
        segment_count += 1

    # With ~10s of audio and 5s segments, we expect at least 1 segment
    assert segment_count >= 1, f"Expected at least 1 audio segment, got {segment_count}"

    # Now test through the async engine with streaming input
    sampling_params = SamplingParams(temperature=0.0, max_tokens=64)

    async def audio_stream():
        for chunk in audio_chunks:
            yield np.array(chunk, dtype=np.float32)

    import asyncio

    input_stream: asyncio.Queue[list[int]] = asyncio.Queue()

    prompt_gen = GraniteSpeechRealtimeGeneration.buffer_realtime_audio(
        audio_stream=audio_stream(),
        input_stream=input_stream,
        model_config=async_engine.model_config,
    )

    full_text = ""
    segment_idx = 0
    async for prompt in prompt_gen:
        request_id = f"granite-rt-{segment_idx}"
        async for resp in async_engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            pass
        # Collect final output for this segment
        full_text += resp.outputs[0].text
        segment_idx += 1

    assert segment_idx >= 1, f"Expected at least 1 segment, got {segment_idx}"
    # Verify transcription contains expected content
    assert len(full_text) > 0, "Expected non-empty transcription"
