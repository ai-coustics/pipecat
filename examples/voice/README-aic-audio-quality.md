# Using Tyto quality scores in a conversation

`voice-aicoustics-audio-quality.py` logs Tyto scores and sustained quality-state
changes without requiring STT, LLM, or TTS services. The policy uses an exponential
moving average (alpha 0.3), three consecutive results, a degradation threshold of
0.5, and a recovery threshold of 0.3. Calibrate these defaults against your calls.
Tyto results cover five-second windows, so this policy adapts to ongoing conditions
rather than individual words or interruptions.

In a conversational bot, place the policy before the user context aggregator:

```python
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.aic_audio_quality_policy import (
    AICAudioQualityPolicy,
    AICAudioQualityPolicyParams,
)

# Use the same normal settings to construct your VAD analyzer.
normal_vad = VADParams(stop_secs=0.2)
policy = AICAudioQualityPolicy(
    analyzer_name=tyto.name,
    params=AICAudioQualityPolicyParams(
        normal_vad_params=normal_vad,
        degraded_vad_params=normal_vad.model_copy(update={"stop_secs": 0.4}),
    ),
)

pipeline = Pipeline([
    transport.input(),
    tyto,
    policy,
    stt,
    user_aggregator,
    llm,
    tts,
    transport.output(),
    assistant_aggregator,
])
```

The VAD profiles above are illustrative and opt-in. Omit both profiles to leave
VAD unchanged. A policy with profiles owns the complete VAD configuration while
active; avoid competing runtime updates from another component.

On sustained degradation, `LLMMessagesTransformFrame` installs one developer
message asking the LLM to clarify ambiguous input and confirm important details.
Recovery removes that message, preserving other context. Neither change starts
an LLM response. `VADParamsUpdateFrame` applies the degraded profile and restores
the normal profile on recovery. Scores continue through the pipeline as ordinary
`MetricsFrame` objects for observers and RTVI clients.

`policy.state` exposes the current smoothed risk and degradation state. Subscribe
to `on_audio_quality_changed` to update application UI or logging. If the audio
source changes or analysis becomes unavailable, call `await policy.reset()` to
remove the note, restore the normal VAD profile, and clear smoothing history.
