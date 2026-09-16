#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Behavior tests for quality-driven context and VAD adaptation."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMContextFrame,
    LLMMessagesTransformFrame,
    MetricsFrame,
    TextFrame,
    VADParamsUpdateFrame,
)
from pipecat.metrics.metrics import AICAudioQualityMetricsData, ProcessingMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext, LLMSpecificMessage
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.aic_audio_quality_policy import (
    AICAudioQualityPolicy,
    AICAudioQualityPolicyParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test


def scores(risk: float, *, processor: str = "tyto", sequence: int = 0):
    return AICAudioQualityMetricsData(
        processor=processor,
        risk_score=risk,
        noise=0.0,
        speaker_reverb=0.0,
        speaker_loudness=1.0,
        interfering_speech=0.0,
        codec_degradation=0.0,
        packet_loss=0.0,
        sequence=sequence,
    )


class TestAudioQualityPolicy(unittest.IsolatedAsyncioTestCase):
    def make_policy(self, **kwargs):
        policy = AICAudioQualityPolicy(
            analyzer_name="tyto", params=AICAudioQualityPolicyParams(**kwargs)
        )
        policy.push_frame = AsyncMock()
        self.addAsyncCleanup(policy.cleanup)
        return policy

    def outputs(self, policy, cls):
        return [
            call.args[0]
            for call in policy.push_frame.await_args_list
            if isinstance(call.args[0], cls)
        ]

    async def test_spikes_and_threshold_hysteresis(self):
        policy = self.make_policy(ema_alpha=1, consecutive_windows=2)
        for risk in [0.9, 0.1, 0.9, 0.4, 0.9]:
            await policy._handle_scores(scores(risk))
        self.assertFalse(policy.state.degraded)
        policy.push_frame.assert_not_called()
        await policy._handle_scores(scores(0.9))
        self.assertTrue(policy.state.degraded)
        for risk in [0.4, 0.2, 0.4, 0.2]:
            await policy._handle_scores(scores(risk))
        self.assertTrue(policy.state.degraded)
        self.assertEqual(len(self.outputs(policy, LLMMessagesTransformFrame)), 1)
        await policy._handle_scores(scores(0.2))
        self.assertFalse(policy.state.degraded)
        self.assertEqual(len(self.outputs(policy, LLMMessagesTransformFrame)), 2)

    async def test_ema_and_neutral_loudness(self):
        policy = self.make_policy(ema_alpha=0.3, consecutive_windows=1)
        await policy._handle_scores(scores(0.0))
        await policy._handle_scores(scores(1.0))
        self.assertAlmostEqual(policy.state.risk_score, 0.3)
        self.assertFalse(policy.state.degraded)
        await policy._handle_scores(scores(1.0))
        self.assertAlmostEqual(policy.state.risk_score, 0.51)
        self.assertTrue(policy.state.degraded)

    async def test_context_is_single_note_and_preserves_other_messages(self):
        policy = self.make_policy(ema_alpha=1, consecutive_windows=1)
        await policy._handle_scores(scores(0.9))
        enter = self.outputs(policy, LLMMessagesTransformFrame)[0]
        original = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "My account number is 123"},
            {"role": "developer", "content": "Keep answers brief"},
            LLMSpecificMessage(llm="test", message={"custom": True}),
        ]
        messages = enter.transform(original)
        self.assertEqual(len(messages), len(original) + 1)
        self.assertEqual(enter.transform(messages), messages)
        self.assertEqual(messages[:-1], original)
        self.assertFalse(enter.run_llm)
        await policy._handle_scores(scores(0.1))
        recovery = self.outputs(policy, LLMMessagesTransformFrame)[1]
        self.assertEqual(recovery.transform(messages), original)
        self.assertFalse(recovery.run_llm)
        # Queued transforms carry their own transition snapshot.
        self.assertEqual(enter.transform(original), messages)

    async def test_vad_profiles_and_reset_restore_normal_settings(self):
        normal = VADParams(confidence=0.65, stop_secs=0.2, min_volume=0.4)
        degraded = normal.model_copy(update={"stop_secs": 0.5})
        policy = self.make_policy(
            ema_alpha=1,
            consecutive_windows=1,
            normal_vad_params=normal,
            degraded_vad_params=degraded,
        )
        events = []

        @policy.event_handler("on_audio_quality_changed")
        async def changed(_policy, state):
            events.append(state)

        await policy._handle_scores(scores(0.9))
        await policy._handle_scores(scores(0.9))
        await policy.reset()
        await asyncio.sleep(0)
        updates = self.outputs(policy, VADParamsUpdateFrame)
        self.assertEqual([frame.params for frame in updates], [degraded, normal])
        self.assertEqual([state.degraded for state in events], [True, False])
        self.assertIsNone(events[-1].risk_score)
        self.assertEqual(updates[0].params.confidence, normal.confidence)
        self.assertEqual(updates[0].params.min_volume, normal.min_volume)
        self.assertIsNot(updates[0].params, degraded)

    async def test_ignores_duplicates_and_invalid_risk(self):
        policy = self.make_policy(ema_alpha=1, consecutive_windows=2)
        await policy._handle_scores(scores(0.9, sequence=1))
        await policy._handle_scores(scores(0.9, sequence=1))
        self.assertFalse(policy.state.degraded)
        for risk in [float("nan"), float("inf"), -0.1, 1.1]:
            await policy._handle_scores(scores(risk))
        await policy._handle_scores(scores(0.9, sequence=2))
        self.assertFalse(policy.state.degraded)
        await policy._handle_scores(scores(0.9, sequence=3))
        self.assertTrue(policy.state.degraded)

    async def test_source_filter_direction_and_passthrough(self):
        policy = self.make_policy(ema_alpha=1, consecutive_windows=1)
        unrelated = MetricsFrame(data=[ProcessingMetricsData(processor="stt", value=1)])
        other_source = MetricsFrame(data=[scores(0.9, processor="other")])
        upstream = MetricsFrame(data=[scores(0.9)])
        with patch.object(FrameProcessor, "process_frame", new_callable=AsyncMock):
            await policy.process_frame(unrelated, FrameDirection.DOWNSTREAM)
            await policy.process_frame(other_source, FrameDirection.DOWNSTREAM)
            await policy.process_frame(upstream, FrameDirection.UPSTREAM)
            self.assertFalse(policy.state.degraded)
            self.assertEqual(policy.push_frame.await_count, 3)
            policy.push_frame.assert_any_await(upstream, FrameDirection.UPSTREAM)
            metric = MetricsFrame(data=[scores(0.9)])
            await policy.process_frame(metric, FrameDirection.DOWNSTREAM)
            self.assertTrue(policy.state.degraded)
            interruption = InterruptionFrame()
            await policy.process_frame(interruption, FrameDirection.DOWNSTREAM)
            self.assertTrue(policy.state.degraded)
            policy.push_frame.assert_any_await(metric, FrameDirection.DOWNSTREAM)

    async def test_observation_only_policy_emits_no_control_frames(self):
        policy = self.make_policy(ema_alpha=1, consecutive_windows=1, update_context=False)
        await policy._handle_scores(scores(0.9))
        self.assertTrue(policy.state.degraded)
        policy.push_frame.assert_not_called()

    def test_invalid_configuration(self):
        for kwargs in [
            {"ema_alpha": 0},
            {"ema_alpha": float("nan")},
            {"consecutive_windows": 0},
            {"degraded_threshold": float("inf")},
            {"degraded_threshold": 0.2, "recovery_threshold": 0.3},
            {"normal_vad_params": VADParams()},
            {"degraded_vad_params": VADParams()},
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AICAudioQualityPolicyParams(**kwargs)

    async def test_real_aggregator_applies_context_and_vad_without_llm_call(self):
        class SilentVAD(VADAnalyzer):
            def num_frames_required(self) -> int:
                return 160

            def voice_confidence(self, buffer: bytes) -> float:
                return 0.0

        normal = VADParams(stop_secs=0.2)
        degraded = normal.model_copy(update={"stop_secs": 0.5})
        vad = SilentVAD(params=normal)
        context = LLMContext(messages=[{"role": "system", "content": "Be helpful"}])
        user = LLMUserAggregator(context, params=LLMUserAggregatorParams(vad_analyzer=vad))
        policy = AICAudioQualityPolicy(
            analyzer_name="tyto",
            params=AICAudioQualityPolicyParams(
                ema_alpha=1,
                consecutive_windows=1,
                normal_vad_params=normal,
                degraded_vad_params=degraded,
            ),
        )
        snapshots = []

        class Capture(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, TextFrame):
                    snapshots.append((list(context.messages), vad.params.model_copy()))
                await self.push_frame(frame, direction)

        received, _ = await run_test(
            Pipeline([policy, user, Capture()]),
            frames_to_send=[
                MetricsFrame(data=[scores(0.9)]),
                SleepFrame(sleep=0.02),
                TextFrame("bad"),
                SleepFrame(sleep=0.02),
                MetricsFrame(data=[scores(0.1)]),
                SleepFrame(sleep=0.02),
                TextFrame("good"),
                SleepFrame(sleep=0.02),
            ],
        )
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(len(snapshots[0][0]), 2)
        self.assertEqual(snapshots[0][1], degraded)
        self.assertEqual(snapshots[1][0], [{"role": "system", "content": "Be helpful"}])
        self.assertEqual(snapshots[1][1], normal)
        self.assertFalse(any(isinstance(frame, LLMContextFrame) for frame in received))
