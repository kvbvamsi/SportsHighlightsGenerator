# agents/audio_agent.py

import os
import yaml
import librosa
import numpy as np
import whisper

from typing import List

from models.schemas import AudioData


class AudioAgent:

    def __init__(self, config_path="configs/settings.yaml"):

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_name = self.config["audio"]["whisper_model"]
        self.sample_rate = self.config["audio"]["sample_rate"]

        print(f"Loading Whisper model: {self.model_name}")

        self.whisper_model = whisper.load_model(
            self.model_name
        )

    def transcribe_audio(
        self,
        video_path: str
    ):

        result = self.whisper_model.transcribe(
            video_path,
            verbose=False
        )

        return result

    def compute_crowd_intensity(
        self,
        audio_signal,
        sr
    ):

        rms = librosa.feature.rms(
            y=audio_signal
        )[0]

        max_rms = np.max(rms)

        if max_rms == 0:
            return 0.0

        normalized = np.mean(rms) / max_rms

        return float(
            min(normalized, 1.0)
        )

    def analyze_audio(
        self,
        video_path: str
    ) -> List[AudioData]:

        transcript = self.transcribe_audio(
            video_path
        )

        audio_signal, sr = librosa.load(
            video_path,
            sr=self.sample_rate
        )

        results = []

        for segment in transcript["segments"]:

            start_time = segment["start"]
            end_time = segment["end"]

            start_idx = int(
                start_time * sr
            )

            end_idx = int(
                end_time * sr
            )

            audio_chunk = audio_signal[
                start_idx:end_idx
            ]

            if len(audio_chunk) == 0:
                crowd_score = 0.0
            else:
                crowd_score = (
                    self.compute_crowd_intensity(
                        audio_chunk,
                        sr
                    )
                )

            results.append(
                AudioData(
                    timestamp=start_time,
                    commentary_text=segment["text"].strip(),
                    crowd_intensity=crowd_score
                )
            )

        return results

    def get_high_energy_segments(
        self,
        audio_results: List[AudioData],
        threshold: float = 0.7
    ):

        return [
            item
            for item in audio_results
            if item.crowd_intensity >= threshold
        ]


audio_agent = AudioAgent()