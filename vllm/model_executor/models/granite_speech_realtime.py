# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only granite-speech realtime model."""

import asyncio
from collections.abc import AsyncGenerator, Mapping

import numpy as np
import torch

from vllm.compilation.decorators import support_torch_compile
from vllm.config import ModelConfig, SpeechToTextConfig, VllmConfig
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.logger import init_logger
from vllm.model_executor.models.granite_speech import (
    GraniteSpeechDummyInputsBuilder,
    GraniteSpeechForConditionalGeneration,
    GraniteSpeechMultiModalProcessingInfo,
    GraniteSpeechMultiModalProcessor,
)
from vllm.model_executor.models.interfaces import (
    SupportsRealtime,
)
from vllm.model_executor.models.qwen3_asr_realtime import (
    Qwen3ASRRealtimeBuffer as RealtimeAudioBuffer,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import _I, BaseMultiModalProcessorCache
from vllm.multimodal.inputs import MultiModalKwargsOptionalItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import BaseDummyInputsBuilder
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.transformers_utils.processor import cached_processor_from_config

logger = init_logger(__name__)


class GraniteSpeechRealtimeMultiModalProcessor(GraniteSpeechMultiModalProcessor):
    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=None)

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], Mapping[str, list[PlaceholderFeaturesInfo]]]:
        audios = mm_kwargs.get("audio", [])
        assert len(audios) == 1, (
            f"Expected only one audio input for realtime, got {len(audios)}"
        )

        audio_data = audios[0]
        audio_embed_size = audio_data.get("audio_embed_sizes")
        if audio_embed_size is not None:
            if isinstance(audio_embed_size.data, torch.Tensor):
                audio_len = int(audio_embed_size.data.item())
            else:
                audio_len = int(audio_embed_size.data)
        else:
            audio_len = 0

        # Get audio_pad token ID and expand placeholder in prompt_ids
        # so that position computation matches seq_len.
        tokenizer = self.info.get_tokenizer()
        audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

        # Find the audio_pad token position and expand it to audio_len tokens
        expanded_ids = list[int]()
        pad_start_idx = -1
        for i, tid in enumerate(prompt_ids):
            if tid == audio_pad_id and pad_start_idx == -1:
                pad_start_idx = i
                expanded_ids.extend([audio_pad_id] * audio_len)
            else:
                expanded_ids.append(tid)

        if pad_start_idx == -1:
            pad_start_idx = 0

        features_info = PlaceholderFeaturesInfo(
            modality="audio",
            item_idx=0,
            start_idx=pad_start_idx,
            tokens=audio_len * [audio_pad_id],
            is_embed=None,
        )
        return expanded_ids, {"audio": [features_info]}


# NOTE: A separate model class is required here because the multimodal
# processor registry binds one processor per model class. The realtime
# endpoint needs a different processor (GraniteSpeechRealtimeMultiModalProcessor)
# than the base transcription endpoint, so we register it on this subclass.
@MULTIMODAL_REGISTRY.register_processor(
    GraniteSpeechRealtimeMultiModalProcessor,
    info=GraniteSpeechMultiModalProcessingInfo,
    dummy_inputs=GraniteSpeechDummyInputsBuilder,
)
@support_torch_compile
class GraniteSpeechRealtimeGeneration(
    GraniteSpeechForConditionalGeneration, SupportsRealtime
):
    realtime_max_tokens = 64

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarray, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.audio_processor
        sampling_rate = feature_extractor.sampling_rate

        # Use a small segment size for low-latency streaming.
        segment_duration_s = 5.0
        buffer = RealtimeAudioBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=segment_duration_s,
        )

        audio_tok = cls.get_placeholder_str("audio", 0)

        # NOTE: only transcription task is supported at this time
        user_prompt = f"{audio_tok}can you transcribe the speech into a written format?"

        tokenizer = cached_tokenizer_from_config(model_config)
        chat = [dict(role="user", content=user_prompt)]
        prompt = tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

        prompt_token_ids = tokenizer.encode(prompt)

        async for audio_chunk in audio_stream:
            buffer.write_audio(audio_chunk)

            while (segment := buffer.read_audio()) is not None:
                yield TokensPrompt(
                    prompt_token_ids=prompt_token_ids,
                    multi_modal_data={"audio": segment},
                )

        remaining = buffer.flush()
        if remaining is not None and len(remaining) > 0:
            yield TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": remaining},
            )

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.audio_processor
        return SpeechToTextConfig(
            max_audio_clip_s=None,
            sample_rate=feature_extractor.sampling_rate,
            min_energy_split_window_size=None,
        )
