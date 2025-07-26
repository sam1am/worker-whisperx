# src/predict.py

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
import copy
import tempfile
import soundfile as sf
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
from speechbrain.inference.speaker import EncoderClassifier, SpeakerRecognition
from sklearn.cluster import AgglomerativeClustering

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
    speaker_verification_model = None

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
        
        # Load the speaker embedding model (for ecapa_tdnn and ahc)
        self.speaker_model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_models",
            run_opts={"device": self.device}
        )

        # Load the speaker verification model (for speechbrain_verify)
        self.speaker_verification_model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_verify_models",
            run_opts={"device": self.device}
        )


    def _diarize_iterative(self, result: Dict, audio_data: np.ndarray, similarity_threshold: float) -> Dict:
        """
        Diarize using the original iterative/greedy speaker embedding comparison.
        """
        result_copy = copy.deepcopy(result)
        known_speakers = []
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)

        for i, segment in enumerate(result_copy['segments']):
            start_time = segment['start']
            end_time = segment.get('end', result_copy['segments'][i+1]['start'] if i + 1 < len(result_copy['segments']) else len(audio_data) / 16000)

            audio_chunk = audio_data[int(start_time * 16000):int(end_time * 16000)]
            if len(audio_chunk) < 100:
                continue

            signal = torch.tensor(audio_chunk).unsqueeze(0)
            embedding = self.speaker_model.encode_batch(signal.to(self.device))
            
            best_match_score = 0
            best_match_speaker = None

            for speaker_id, speaker_embedding in known_speakers:
                score = cosine_sim(embedding, speaker_embedding).item()
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
        
        # Assign speakers to the flat word list
        word_idx = 0
        for segment in result_copy.get('segments', []):
            speaker = segment.get('speaker')
            if speaker and 'words' in segment:
                for _ in segment['words']:
                    if word_idx < len(result_copy['word_segments']):
                        result_copy['word_segments'][word_idx]['speaker'] = speaker
                    word_idx += 1
        
        return result_copy

    def _diarize_ahc(self, result: Dict, audio_data: np.ndarray, distance_threshold: float) -> Dict:
        """
        Diarize using Agglomerative Hierarchical Clustering.
        """
        result_copy = copy.deepcopy(result)
        segments = result_copy['segments']
        all_embeddings = []

        for i, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment.get('end', segments[i+1]['start'] if i + 1 < len(segments) else len(audio_data) / 16000)
            audio_chunk = audio_data[int(start_time * 16000):int(end_time * 16000)]
            
            if len(audio_chunk) < 100:
                all_embeddings.append(None)
                continue

            signal = torch.tensor(audio_chunk).unsqueeze(0)
            embedding = self.speaker_model.encode_batch(signal.to(self.device))
            all_embeddings.append(embedding.squeeze().cpu().numpy())

        valid_embeddings = [(i, emb) for i, emb in enumerate(all_embeddings) if emb is not None]
        if not valid_embeddings:
            return result_copy

        segment_map, embeddings_np = zip(*valid_embeddings)
        embeddings_np = np.array(embeddings_np)
        
        if len(embeddings_np) == 1:
            speaker_id = "SPEAKER_00"
            original_segment_index = segment_map[0]
            segments[original_segment_index]['speaker'] = speaker_id
            if 'words' in segments[original_segment_index]:
                for word in segments[original_segment_index]['words']:
                    word['speaker'] = speaker_id
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None, affinity='cosine', linkage='average', distance_threshold=distance_threshold
            ).fit(embeddings_np)
            speaker_labels = clustering.labels_
            for i, label in enumerate(speaker_labels):
                original_segment_index = segment_map[i]
                speaker_id = f"SPEAKER_{label:02d}"
                segments[original_segment_index]['speaker'] = speaker_id
                if 'words' in segments[original_segment_index]:
                    for word in segments[original_segment_index]['words']:
                        word['speaker'] = speaker_id
        
        # Re-create the flat word list with speaker assignments
        new_word_segments = []
        for segment in result_copy.get('segments', []):
            if 'speaker' in segment and 'words' in segment:
                for word in segment['words']:
                    word_with_speaker = word.copy()
                    word_with_speaker['speaker'] = segment['speaker']
                    new_word_segments.append(word_with_speaker)
        result_copy['word_segments'] = new_word_segments
        
        return result_copy

    def _diarize_speechbrain_verify(self, result: Dict, audio_data: np.ndarray) -> Dict:
        """
        Diarize using the high-level SpeechBrain SpeakerRecognition class.
        This is an iterative approach that requires writing temp files.
        """
        result_copy = copy.deepcopy(result)
        known_speakers = [] # List of tuples: (speaker_id, temp_file_path)

        # Create a temporary directory to store audio chunks
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, segment in enumerate(result_copy['segments']):
                start_time = segment['start']
                end_time = segment.get('end', result_copy['segments'][i+1]['start'] if i + 1 < len(result_copy['segments']) else len(audio_data) / 16000)

                audio_chunk = audio_data[int(start_time * 16000):int(end_time * 16000)]
                if len(audio_chunk) < 100:
                    continue
                
                # Write current chunk to a temp file
                current_chunk_path = os.path.join(temp_dir, f"segment_{i}.wav")
                sf.write(current_chunk_path, audio_chunk, 16000)

                best_match_speaker = None
                for speaker_id, speaker_chunk_path in known_speakers:
                    # score is a tensor, prediction is a tensor (1 for same, 0 for different)
                    score_tensor, prediction_tensor = self.speaker_verification_model.verify_files(
                        speaker_chunk_path, current_chunk_path
                    )
                    if prediction_tensor.item() == 1:
                        best_match_speaker = speaker_id
                        break # Found a match, no need to check other known speakers
                
                if best_match_speaker:
                    assigned_speaker = best_match_speaker
                else:
                    assigned_speaker = f"SPEAKER_{len(known_speakers):02d}"
                    known_speakers.append((assigned_speaker, current_chunk_path))
                
                segment['speaker'] = assigned_speaker
                if 'words' in segment:
                    for word in segment['words']:
                        word['speaker'] = assigned_speaker
        
        # Re-create the flat word list with speaker assignments
        new_word_segments = []
        for segment in result_copy.get('segments', []):
            if 'speaker' in segment and 'words' in segment:
                for word in segment['words']:
                    word_with_speaker = word.copy()
                    word_with_speaker['speaker'] = segment['speaker']
                    new_word_segments.append(word_with_speaker)
        result_copy['word_segments'] = new_word_segments

        return result_copy


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
        similarity_threshold: float = 0.65,
        distance_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Run a single prediction on the model
        """
        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")

        audio_data = load_audio(audio)
        
        # Calculate audio duration
        duration_in_seconds = len(audio_data) / 16000.0

        result = model.transcribe(
            audio_data, batch_size=batch_size, language=language, print_progress=True)
        
        # Add audio_duration to the result
        result['audio_duration'] = duration_in_seconds

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
                    result = self._diarize_iterative(result, audio_data, similarity_threshold)
                elif diarization_method == "ahc":
                    result = self._diarize_ahc(result, audio_data, distance_threshold)
                elif diarization_method == "speechbrain_verify":
                    result = self._diarize_speechbrain_verify(result, audio_data)
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

                result = make_serializable(result)

            except Exception as e:
                result["diarization_error"] = str(e)
                import traceback
                result["diarization_traceback"] = traceback.format_exc()

        return result