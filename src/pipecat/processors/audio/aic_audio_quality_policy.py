#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Translate sustained Tyto audio-quality changes into Pipecat context and VAD frames."""

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMMessagesTransformFrame,
    MetricsFrame,
    StartFrame,
    StopFrame,
    VADParamsUpdateFrame,
)
from pipecat.metrics.metrics import AICAudioQualityMetricsData
from pipecat.processors.aggregators.llm_context import LLMContextMessage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class AICAudioQualityPolicyParams(BaseModel):
    """Settings for sustained audio-quality transitions.

    Parameters:
        ema_alpha: Weight of each new risk score in the exponential moving average.
        degraded_threshold: Smoothed risk at or above which degradation is counted.
        recovery_threshold: Smoothed risk at or below which recovery is counted.
        consecutive_windows: Consecutive qualifying results needed for a transition.
            Results may describe overlapping windows; this is not a duration.
        update_context: Maintain one developer message describing degraded audio.
        normal_vad_params: Application's normal VAD settings, restored on recovery.
        degraded_vad_params: Application's VAD settings for degraded audio. Both VAD
            profiles must be provided to enable adaptation. These replace all VAD
            parameters, so this policy should be their sole runtime owner.
    """

    ema_alpha: float = Field(default=0.3, gt=0, le=1, allow_inf_nan=False)
    degraded_threshold: float = Field(default=0.5, ge=0, le=1, allow_inf_nan=False)
    recovery_threshold: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    consecutive_windows: int = Field(default=3, ge=1)
    update_context: bool = True
    normal_vad_params: VADParams | None = None
    degraded_vad_params: VADParams | None = None

    @model_validator(mode="after")
    def _validate_policy(self) -> Self:
        if self.recovery_threshold >= self.degraded_threshold:
            raise ValueError("recovery_threshold must be below degraded_threshold")
        if (self.normal_vad_params is None) != (self.degraded_vad_params is None):
            raise ValueError("Provide both normal_vad_params and degraded_vad_params")
        return self


class AICAudioQualityState(BaseModel):
    """Snapshot emitted by an audio-quality policy transition.

    Parameters:
        degraded: Whether the audio is persistently degraded.
        risk_score: Smoothed risk at the transition, or None after an explicit reset.
    """

    model_config = ConfigDict(frozen=True)

    degraded: bool
    risk_score: float | None


class AICAudioQualityPolicy(FrameProcessor):
    """Consume Tyto metrics and adapt context and optional VAD settings.

    Place this processor after ``AICTytoAnalyzer`` and before the user context
    aggregator. It forwards every input frame unchanged. Only downstream Tyto
    metrics from ``analyzer_name`` affect the policy. Thresholds apply to an EMA
    of risk, with separate degradation/recovery thresholds and consecutive-result
    gating to prevent rapid toggling.

    On degradation, a context transform installs one developer message. Recovery
    removes it. Neither update triggers an LLM response. Optional VAD profiles
    are sent downstream to the user aggregator's VAD controller on transitions.
    Ordinary interruptions preserve the quality state; start/end/stop and
    :meth:`reset` clear it. Applications should call ``reset`` after a source
    discontinuity or if analysis becomes unavailable.

    Event handlers:

    - on_audio_quality_changed: Called with an :class:`AICAudioQualityState` on
      degradation, recovery, or a reset of degraded state.
    """

    def __init__(
        self,
        *,
        analyzer_name: str,
        params: AICAudioQualityPolicyParams | None = None,
        **kwargs,
    ) -> None:
        """Initialize the policy.

        Args:
            analyzer_name: Name of the Tyto processor whose metrics to consume.
            params: Smoothing, transition, context, and optional VAD settings.
            **kwargs: Additional arguments passed to FrameProcessor.
        """
        super().__init__(**kwargs)
        self._analyzer_name = analyzer_name
        self._params = (params or AICAudioQualityPolicyParams()).model_copy(deep=True)
        self._risk: float | None = None
        self._degraded = False
        self._consecutive = 0
        self._last_sequence = 0
        self._context_prefix = f"[Audio quality policy: {self.name}]\n"
        self._register_event_handler("on_audio_quality_changed")

    @property
    def state(self) -> AICAudioQualityState:
        """Return the current degradation state and smoothed risk."""
        return AICAudioQualityState(degraded=self._degraded, risk_score=self._risk)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Forward frames and apply qualifying downstream audio-quality metrics."""
        await super().process_frame(frame, direction)
        if direction == FrameDirection.DOWNSTREAM and isinstance(frame, (EndFrame, StopFrame)):
            await self.reset()
        await self.push_frame(frame, direction)
        if direction != FrameDirection.DOWNSTREAM:
            return
        if isinstance(frame, StartFrame):
            await self.reset()
        elif isinstance(frame, MetricsFrame):
            for data in frame.data:
                if (
                    isinstance(data, AICAudioQualityMetricsData)
                    and data.processor == self._analyzer_name
                ):
                    await self._handle_scores(data)

    async def reset(self) -> None:
        """Clear history, remove the quality note, and restore normal VAD settings."""
        was_degraded = self._degraded
        self._risk = None
        self._degraded = False
        self._consecutive = 0
        self._last_sequence = 0
        if was_degraded:
            await self._publish_transition()

    async def _handle_scores(self, scores: AICAudioQualityMetricsData) -> None:
        if not math.isfinite(scores.risk_score) or not 0 <= scores.risk_score <= 1:
            self._consecutive = 0
            return
        if scores.sequence > 0:
            if scores.sequence <= self._last_sequence:
                return
            self._last_sequence = scores.sequence

        alpha = self._params.ema_alpha
        self._risk = (
            scores.risk_score
            if self._risk is None
            else alpha * scores.risk_score + (1 - alpha) * self._risk
        )
        qualifies = (
            self._risk <= self._params.recovery_threshold
            if self._degraded
            else self._risk >= self._params.degraded_threshold
        )
        self._consecutive = self._consecutive + 1 if qualifies else 0
        if self._consecutive >= self._params.consecutive_windows:
            self._degraded = not self._degraded
            self._consecutive = 0
            await self._publish_transition()

    async def _publish_transition(self) -> None:
        if self._params.update_context:
            # Capture the state in the frame: it may be consumed after another transition.
            degraded = self._degraded
            prefix = self._context_prefix

            def transform(messages: list[LLMContextMessage]) -> list[LLMContextMessage]:
                updated: list[LLMContextMessage] = []
                for message in messages:
                    if isinstance(message, dict) and message.get("role") == "developer":
                        content = message.get("content")
                        if isinstance(content, str) and content.startswith(prefix):
                            continue
                    updated.append(message)
                if degraded:
                    updated.append(
                        {
                            "role": "developer",
                            "content": prefix
                            + "The caller's recent audio has persistently elevated audio-quality "
                            "risk. Recognition may be unreliable. If their request is ambiguous, "
                            "ask for clarification and confirm important names or numbers. "
                            "Do not assume a particular cause or mention this measurement "
                            "unless it is relevant to helping the caller.",
                        }
                    )
                return updated

            await self.push_frame(LLMMessagesTransformFrame(transform=transform, run_llm=False))

        vad_params = (
            self._params.degraded_vad_params if self._degraded else self._params.normal_vad_params
        )
        if vad_params is not None:
            await self.push_frame(VADParamsUpdateFrame(params=vad_params.model_copy(deep=True)))
        await self._call_event_handler("on_audio_quality_changed", self.state)
