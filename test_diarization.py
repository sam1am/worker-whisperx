# test_diarization.py

import argparse
import copy
import os
import sys
import tempfile
import time
from typing import Dict, Any

import numpy as np
import requests
import torch
from sklearn.cluster import AgglomerativeClustering

# --- Setup Project Path ---
# This robustly adds the 'src' directory to the Python path
# ensuring that the script can find your 'predict' module.
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
sys.path.insert(0, src_path)

try:
    from predict import Predictor, load_audio, load_align_model, align
except ImportError as e:
    print("Error: Could not import from 'src/predict.py'.")
    print(f"PYTHONPATH issue: {e}")
    print("Please ensure this script is in the root directory of your project.")
    sys.exit(1)


def diarize_iterative(
    result: Dict[str, Any],
    audio_data: np.ndarray,
    speaker_model,
    device: str,
    similarity_threshold: float = 0.65
) -> Dict[str, Any]:
    """
    Performs diarization using your original greedy/iterative approach.
    """
    print(f"\n[Iterative] Running with similarity_threshold = {similarity_threshold}")
    start_time = time.time()

    known_speakers = []
    cosine_sim = torch.nn.CosineSimilarity(dim=-1)

    # Use a deep copy to avoid modifying the original result object
    result = copy.deepcopy(result)

    for i, segment in enumerate(result['segments']):
        start_time_seg = segment['start']
        # Ensure 'end' exists, falling back to the next segment's start or audio duration
        end_time_seg = segment.get('end', result['segments'][i+1]['start'] if i + 1 < len(result['segments']) else len(audio_data) / 16000)

        audio_chunk = audio_data[int(start_time_seg * 16000):int(end_time_seg * 16000)]
        if len(audio_chunk) == 0:
            continue

        signal = torch.tensor(audio_chunk).unsqueeze(0)
        embedding = speaker_model.encode_batch(signal.to(device))

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

    end_time = time.time()
    print(f"[Iterative] Processing took {end_time - start_time:.2f} seconds.")
    return result


def diarize_ahc(
    result: Dict[str, Any],
    audio_data: np.ndarray,
    speaker_model,
    device: str,
    distance_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Performs diarization using Agglomerative Hierarchical Clustering.
    """
    print(f"\n[AHC] Running with distance_threshold = {distance_threshold}")
    start_time = time.time()

    # Use a deep copy to avoid modifying the original result object
    result = copy.deepcopy(result)
    segments = result['segments']
    all_embeddings = []

    # 1. Extract embeddings for all segments
    for i, segment in enumerate(segments):
        start_time_seg = segment['start']
        end_time_seg = segment.get('end', segments[i+1]['start'] if i + 1 < len(segments) else len(audio_data) / 16000)

        audio_chunk = audio_data[int(start_time_seg * 16000):int(end_time_seg * 16000)]
        if len(audio_chunk) < 100:  # Skip very short segments
            all_embeddings.append(None) # Keep a placeholder
            continue

        signal = torch.tensor(audio_chunk).unsqueeze(0)
        embedding = speaker_model.encode_batch(signal.to(device))
        all_embeddings.append(embedding.squeeze().cpu().numpy())
    
    # Filter out any segments that failed to produce an embedding
    valid_embeddings_np = np.array([e for e in all_embeddings if e is not None])
    segment_map = [i for i, e in enumerate(all_embeddings) if e is not None]

    if len(valid_embeddings_np) == 0:
        print("[AHC] No valid embeddings found to cluster.")
        return result
    
    if len(valid_embeddings_np) == 1:
        print("[AHC] Only one valid segment found.")
        segments[segment_map[0]]['speaker'] = "SPEAKER_00"
        return result

    # 2. Perform Agglomerative Clustering
    # The distance_threshold is the key parameter to tune.
    # It's the distance at which the algorithm stops merging clusters.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        affinity='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    ).fit(valid_embeddings_np)

    speaker_labels = clustering.labels_

    # 3. Assign speaker labels back to segments
    for i, label in enumerate(speaker_labels):
        original_segment_index = segment_map[i]
        speaker_id = f"SPEAKER_{label:02d}"
        segments[original_segment_index]['speaker'] = speaker_id
        if 'words' in segments[original_segment_index]:
            for word in segments[original_segment_index]['words']:
                word['speaker'] = speaker_id
    
    end_time = time.time()
    print(f"[AHC] Processing took {end_time - start_time:.2f} seconds.")
    return result


def print_report(method_name: str, result: Dict[str, Any], num_segments_to_show: int = 15):
    """Prints a summary report of the diarization result."""
    speakers = set()
    for segment in result.get('segments', []):
        if 'speaker' in segment:
            speakers.add(segment['speaker'])

    print(f"\n--- REPORT: {method_name.upper()} ---")
    print(f"Total Speakers Detected: {len(speakers)}")
    print("-" * (len(method_name) + 15))

    for i, segment in enumerate(result.get('segments', [])):
        if i >= num_segments_to_show:
            print("...")
            break
        
        speaker = segment.get('speaker', 'UNKNOWN')
        start = segment.get('start', 0)
        end = segment.get('end', 0)
        text = segment.get('text', '').strip()
        print(f"[{start:06.2f}s - {end:06.2f}s] Speaker: {speaker} | Text: {text}")


def main():
    parser = argparse.ArgumentParser(
        description="Test and compare diarization approaches.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--audio_url',
        required=True,
        type=str,
        help="URL of the audio file to process."
    )
    parser.add_argument(
        '--iterative_threshold',
        type=float,
        default=0.65,
        help="Similarity threshold for the iterative method (higher is stricter)."
    )
    parser.add_argument(
        '--ahc_threshold',
        type=float,
        default=0.5,
        help="Distance threshold for the AHC method (lower is stricter)."
    )
    args = parser.parse_args()

    # --- 1. Setup Models ---
    print("Setting up models, this may take a moment...")
    predictor = Predictor()
    predictor.setup()
    device = predictor.device
    print(f"Models loaded. Using device: {device}")

    # --- 2. Download and Load Audio ---
    print(f"\nDownloading audio from: {args.audio_url}")
    try:
        response = requests.get(args.audio_url, timeout=30)
        response.raise_for_status()
        
        # Use a temporary file to handle the audio data
        # Ensure the suffix matches the likely format or is generic
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as tmp:
            tmp.write(response.content)
            tmp_path = tmp.name
        
        audio_data = load_audio(tmp_path)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading audio: {e}")
        return
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

    # --- 3. Transcribe and Align (once) ---
    print("\nRunning transcription and alignment...")
    # Using a fast model for testing purposes
    whisper_model = predictor.models['tiny']
    transcription_result = whisper_model.transcribe(audio_data, batch_size=16)

    align_model, align_metadata = load_align_model(
        language_code=transcription_result["language"], device=device
    )
    aligned_result = align(
        transcription_result["segments"],
        align_model,
        align_metadata,
        audio_data,
        device,
        return_char_alignments=False
    )
    print("Transcription complete.")

    # --- 4. Run and Compare Diarization Methods ---
    iterative_result = diarize_iterative(
        aligned_result, audio_data, predictor.speaker_model, device, args.iterative_threshold
    )
    
    ahc_result = diarize_ahc(
        aligned_result, audio_data, predictor.speaker_model, device, args.ahc_threshold
    )
    
    # --- 5. Print Reports ---
    print_report("Iterative Method", iterative_result)
    print_report("AHC Method", ahc_result)


if __name__ == "__main__":
    main()