#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AI Coustics audio enhancement filter for Pipecat.

This module provides an audio filter implementation using AI Coustics' RealTimeL
enhancement technology to improve audio quality in real-time streams.
"""

import os

import numpy as np
from loguru import logger

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter
from pipecat.frames.frames import (FilterControlFrame, FilterEnableFrame,
                                   FilterUpdateSettingsFrame)

try:
    from aicoustics import RealTimeL
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use the AI Coustics filter, you need to `pip install aicoustics`.")
    raise Exception(f"Missing module: {e}")


class AiCousticsProcessorManager:
    """Singleton manager for RealTimeL enhancement instances.

    Ensures that only one RealTimeL instance exists for the entire
    program with specific parameters.
    """

    _instances = {}

    @classmethod
    def get_processor(cls, license_key: str, sample_rate: int, num_channels: int, num_frames: int):
        """Get or create a RealTimeL enhancement instance.

        Args:
            sample_rate: Audio sample rate in Hz.
            num_channels: Number of audio channels.
            num_frames: Number of frames per processing chunk.

        Returns:
            Shared RealTimeL instance for the given parameters.
        """
        key = (sample_rate, num_channels, num_frames)
        if key not in cls._instances:
            cls._instances[key] = RealTimeL(
                license_key=license_key,
                num_channels=num_channels, 
                sample_rate=sample_rate, 
                num_frames=num_frames
            )
        return cls._instances[key]


class AiCousticsFilter(BaseAudioFilter):
    """Audio filter using AI Coustics enhancement technology.

    Provides real-time audio enhancement for audio streams using AI Coustics'
    proprietary enhancement algorithms. Automatically adapts to different
    sample rates while maintaining optimal processing chunk sizes.
    """

    def __init__(
        self, 
        license_key: str,
        channels: int = 1, 
        num_frames: int = 512,
        enhancement_strength: float = 1.0
    ) -> None:
        """Initialize the AI Coustics enhancement filter.

        Args:
            channels: Number of audio channels. Defaults to 1 (mono).
            num_frames: Number of frames per processing chunk. Defaults to 512.
            enhancement_strength: Enhancement strength from 0.0 to 1.0. Defaults to 1.0.
        """
        super().__init__()
        
        self._license_key = license_key
        self._channels = channels
        self._num_frames = num_frames
        self._enhancement_strength = max(0.0, min(1.0, enhancement_strength))
        self._sample_rate = 0
        self._filtering = True
        self._enhancer = None
        self._buffer = np.array([], dtype=np.float32)

    async def start(self, sample_rate: int):
        """Initialize the AI Coustics processor with the transport's sample rate.

        Args:
            sample_rate: The sample rate of the input transport in Hz.
        """
        self._sample_rate = sample_rate
        self._enhancer = AiCousticsProcessorManager.get_processor(
            self._license_key,
            self._sample_rate, self._channels, self._num_frames
        )
        
        # Set enhancement strength
        self._enhancer.set_enhancement_strength(self._enhancement_strength)
        
        # Log processor information
        logger.info(f"AI Coustics filter started:")
        logger.info(f"  Sample rate: {self._sample_rate} Hz")
        logger.info(f"  Channels: {self._channels}")
        logger.info(f"  Frames per chunk: {self._num_frames}")
        logger.info(f"  Enhancement strength: {int(self._enhancement_strength * 100)}%")
        logger.info(f"  Optimal input buffer size: {self._enhancer.get_optimal_num_frames()} samples")
        logger.info(f"  Optimal sample rate: {self._enhancer.get_optimal_sample_rate()} Hz")
        logger.info(f"  Current algorithmic latency: {self._enhancer.get_latency()/self._sample_rate * 1000:.2f}ms")

    async def stop(self):
        """Clean up the AI Coustics processor when stopping."""
        self._enhancer = None
        self._buffer = np.array([], dtype=np.float32)

    async def process_frame(self, frame: FilterControlFrame):
        """Process control frames to enable/disable filtering or update settings.

        Args:
            frame: The control frame containing filter commands.
        """
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable
            logger.info(f"AI Coustics filter {'enabled' if frame.enable else 'disabled'}")
        elif isinstance(frame, FilterUpdateSettingsFrame):
            # Handle settings updates from the settings dictionary
            if 'enhancement_strength' in frame.settings:
                old_strength = self._enhancement_strength
                self._enhancement_strength = max(0.0, min(1.0, frame.settings['enhancement_strength']))
                if self._enhancer:
                    self._enhancer.set_enhancement_strength(self._enhancement_strength)
                logger.info(f"AI Coustics enhancement strength updated from {int(old_strength * 100)}% to {int(self._enhancement_strength * 100)}%")

    async def filter(self, audio: bytes) -> bytes:
        """Apply AI Coustics enhancement to audio data.

        Processes audio in chunks using the AI Coustics RealTimeL processor.
        Handles buffering for consistent chunk sizes and maintains audio continuity.

        Args:
            audio: Raw audio data as bytes to be filtered.

        Returns:
            Enhanced audio data as bytes.
        """
        if not self._filtering or not self._enhancer:
            return audio

        # Convert bytes to numpy array
        data = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        
        # Normalize to [-1, 1] range
        data = data / 32768.0
        
        # Reshape for multi-channel processing (channels, samples)
        if self._channels == 1:
            data = data.reshape(1, -1)
        else:
            # For multi-channel, interleaved -> deinterleaved
            data = data.reshape(-1, self._channels).T
        
        # Add to buffer
        if self._buffer.size == 0:
            self._buffer = data.copy()
        else:
            self._buffer = np.concatenate([self._buffer, data], axis=1)
        
        # Process complete chunks
        output_chunks = []
        while self._buffer.shape[1] >= self._num_frames:
            # Extract chunk
            chunk = self._buffer[:, :self._num_frames].copy()
            self._buffer = self._buffer[:, self._num_frames:]
            
            # Create padded chunk for processing
            padded_chunk = np.zeros((self._channels, self._num_frames), dtype=np.float32)
            padded_chunk[:, :chunk.shape[1]] = chunk
            
            # Process with AI Coustics
            self._enhancer.process_deinterleaved(padded_chunk)
            
            # Store processed chunk (only the valid part)
            output_chunks.append(padded_chunk[:, :chunk.shape[1]])
        
        # If no complete chunks were processed, return original audio
        if not output_chunks:
            return audio
            
        # Concatenate processed chunks
        output = np.concatenate(output_chunks, axis=1)
        
        # Convert back to interleaved format for multi-channel
        if self._channels == 1:
            output = output.flatten()
        else:
            output = output.T.flatten()
        
        # Convert back to int16 and clip
        output = np.clip(output * 32768.0, -32768, 32767).astype(np.int16)
        
        return output.tobytes()