"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

import os
import numpy as np
import pandas as pd
import torch
import torchaudio
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
from speechbrain.inference.speaker import EncoderClassifier

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
    speaker_model = None

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
        Load the pyannote diarization model
        """
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
        self.diarize_model = diarize.DiarizationPipeline(
            use_auth_token=hf_token,
            device=self.device
        )

    def setup(self):
        """
        Load the models into memory to make running multiple predictions efficient
        """
        model_names = ["tiny", "large-v2"]
        with ThreadPoolExecutor() as executor:
            for model_name, model in executor.map(self.load_model, model_names):
                if model_name is not None:
                    self.models[model_name] = model
        
        # Load the speaker embedding model
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_models",
            run_opts={"device": self.device}
        )

    def _diarize_ecapa_tdnn(self, result: Dict, audio_data: np.ndarray) -> Dict:
        """
        Diarize using ECAPA-TDNN speaker embeddings.
        """
        known_speakers = []
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity_threshold = 0.65  # Heuristic threshold

        for i, segment in enumerate(result['segments']):
            start_time = segment['start']
            end_time = segment.get('end', result['segments'][i+1]['start'] if i + 1 < len(result['segments']) else len(audio_data) / 16000)

            # Extract audio chunk for the segment
            audio_chunk = audio_data[int(start_time * 16000):int(end_time * 16000)]
            if len(audio_chunk) == 0:
                continue

            signal = torch.tensor(audio_chunk).unsqueeze(0)
            
            # Resample if necessary
            if signal.shape[1] > 16000:
                # This is a simplification; a more robust solution would handle resampling properly
                signal = torchaudio.functional.resample(signal.float(), orig_freq=16000, new_freq=16000)

            embedding = self.speaker_model.encode_batch(signal.to(self.device))
            
            best_match_score = 0
            best_match_speaker = None

            for speaker_id, speaker_embedding in known_speakers:
                score = cosine_sim(embedding, speaker_embedding)
                if score > best_match_score:
                    best_match_score = score
                    best_match_speaker = speaker_id
            
            if best_match_score > similarity_threshold:
                assigned_speaker = best_match_speaker
            else:
                assigned_speaker = f"SPEAKER_{len(known_speakers):02d}"
                known_speakers.append((assigned_speaker, embedding))
            
            segment['speaker'] = assigned_speaker
            if 'words' in segment:
                for word in segment['words']:
                    word['speaker'] = assigned_speaker
        
        # Update word_segments list as well
        word_idx = 0
        for seg in result['segments']:
            if 'words' in seg:
                for _ in seg['words']:
                    result['word_segments'][word_idx]['speaker'] = seg['speaker']
                    word_idx += 1
        
        return result


    def __call__(
        self,
        audio: str,
        model_name: str = "large-v2",
        language: Optional[str] = None,
        batch_size: int = 16,
        diarize: bool = False,
        diarization_method: str = "pyannote",
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
            audio_data, batch_size=batch_size, language=language, print_progress=True)

        model_a, metadata = load_align_model(
            language_code=result["language"], device=self.device)

        result = align(
            result["segments"], model_a, metadata, audio_data, self.device,
            return_char_alignments=False, print_progress=True)
        
        result["word_segments"] = result.pop("word_segments", []) # Ensure key exists

        # Handle diarization if requested
        if diarize:
            try:
                if diarization_method == "ecapa_tdnn":
                    result = self._diarize_ecapa_tdnn(result, audio_data)
                elif diarization_method == "pyannote":
                    if not self.diarize_model:
                        self.load_diarize_model(hf_token)
                    if not self.diarize_model:
                        raise ValueError("Pyannote diarization model could not be loaded.")
                    
                    diarize_segments = self.diarize_model(
                        audio_data, min_speakers=min_speakers, max_speakers=max_speakers)
                    
                    result = assign_word_speakers(diarize_segments, result)
                else:
                    raise ValueError(f"Unknown diarization_method: {diarization_method}")

                # Make sure all elements in the result are JSON serializable
                result = make_serializable(result)

            except Exception as e:
                result["diarization_error"] = str(e)
                import traceback
                result["diarization_traceback"] = traceback.format_exc()

        return result