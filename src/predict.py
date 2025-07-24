"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

import os
import numpy as np
import pandas as pd
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple, Dict, Any
from runpod.serverless.utils import rp_cuda
from whisperx.asr import FasterWhisperPipeline
from whisperx import (
    load_model,
    load_audio,
    load_align_model,
    align,
    diarize,
    assign_word_speakers
)
from rp_schema import *

# Add NAN to numpy namespace to fix compatibility with pyannote.audio
if not hasattr(np, 'NAN'):
    np.NAN = np.nan

# Custom JSON encoder to handle non-serializable objects


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)


def make_serializable(obj):
    """Convert all non-serializable objects to serializable format"""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif pd.isna(obj):
        return None
    else:
        return obj


class Predictor:
    """
    A Predictor class for the WhisperX
    """

    models: Dict[str, FasterWhisperPipeline] = {}
    model_dir: str = "/tmp"
    diarize_model = None

    @property
    def device(self):
        return "cuda" if rp_cuda.is_available() else "cpu"

    @property
    def compute_type(self):
        return "float16" if rp_cuda.is_available() else "int8"

    def load_model(self, model_name) -> Tuple[str, FasterWhisperPipeline]:
        """
        Load the model from the weights folder
        """

        loaded_model = load_model(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=self.model_dir,
            asr_options={
                "max_new_tokens": None,
                "clip_timestamps": None,
                "hallucination_silence_threshold": None,
            }
        )

        return model_name, loaded_model

    def load_diarize_model(self, hf_token: Optional[str] = None):
        """
        Load the diarization model
        """
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        self.diarize_model = diarize.DiarizationPipeline(
            use_auth_token=hf_token,
            device=self.device
        )

    def setup(self):
        """
        Load the model into memory to make running multiple predictions efficient
        """

        # ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
        model_names = ["tiny", "large-v2"]
        with ThreadPoolExecutor() as executor:
            for model_name, model in executor.map(self.load_model, model_names):
                if model_name is not None:
                    self.models[model_name] = model

    def __call__(
        self,
        audio: str,
        model_name: str = "large-v2",
        language: Optional[str] = None,
        batch_size: int = 16,
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        hf_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a single prediction on the model
        """

        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")

        audio_data = load_audio(audio)
        result = model.transcribe(
            audio_data,
            batch_size=batch_size,
            language=language,
            print_progress=True
        )

        model_a, metadata = load_align_model(
            language_code=result["language"],
            device=self.device
        )

        result = align(
            result["segments"],
            model_a,
            metadata,
            audio_data,
            self.device,
            return_char_alignments=False,
            print_progress=True
        )

        # Handle diarization if requested
        if diarize:
            try:
                if not self.diarize_model:
                    self.load_diarize_model(hf_token)

                if not self.diarize_model:
                    raise ValueError(
                        "Diarization model could not be loaded. Make sure to provide a valid HF token.")

                diarize_segments = self.diarize_model(
                    audio_data,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )

                # Convert diarize_segments to a serializable format if it contains a DataFrame
                if hasattr(diarize_segments, 'get') and isinstance(diarize_segments.get('segments', None), pd.DataFrame):
                    diarize_segments['segments'] = diarize_segments['segments'].to_dict(
                        'records')

                result = assign_word_speakers(diarize_segments, result)

                # Make sure all elements in the result are JSON serializable
                result = make_serializable(result)

            except Exception as e:
                # Add error information to the result but continue with transcription
                result["diarization_error"] = str(e)
                import traceback
                result["diarization_traceback"] = traceback.format_exc()

        return result
