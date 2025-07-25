"""
This script pre-loads the diarization model during the Docker build process
to avoid the first-run delay in production.
"""
import os
import sys
import numpy as np
import torch
from faster_whisper import WhisperModel
from whisperx.asr import FasterWhisperPipeline
from whisperx import (
    load_model,
    load_audio,
    diarize
)
from speechbrain.inference.speaker import EncoderClassifier


# Add NAN to numpy namespace to fix compatibility with pyannote.audio
if not hasattr(np, 'NAN'):
    np.NAN = np.nan


def main():
    # Check if HF_TOKEN is provided
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not provided. Diarization model may not load correctly.")

    # Set up device and compute type
    device = "cpu"  # Use CPU during build
    compute_type = "int8"

    print("Loading WhisperX model...")
    # Load a small model just for testing
    model = load_model(
        "tiny",
        device=device,
        compute_type=compute_type,
        download_root="/tmp",
        asr_options={
            "max_new_tokens": None,
            "clip_timestamps": None,
            "hallucination_silence_threshold": None,
        }
    )

    print("Loading pyannote diarization model...")
    # Load the diarization model
    diarize_model = diarize.DiarizationPipeline(
        use_auth_token=hf_token,
        device=device
    )

    # Load sample audio
    print("Loading sample audio...")
    audio_path = "audio/sample.mp3"
    audio_data = load_audio(audio_path)

    # Run a quick diarization to ensure models are cached
    print("Running sample pyannote diarization...")
    try:
        diarize_segments = diarize_model(audio_data, min_speakers=1, max_speakers=2)
        print("Pyannote diarization successful! Models are cached.")
    except Exception as e:
        print(f"Error during pyannote diarization: {e}")
        import traceback
        print(traceback.format_exc())
        print("Pyannote models may not have been properly cached.")
        sys.exit(1)

    # Preload ECAPA-TDNN model
    print("Loading ECAPA-TDNN speaker recognition model...")
    try:
        speaker_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_models",  # Cache in a specific directory
            run_opts={"device": device}
        )
        # Create a dummy tensor to ensure the model is ready
        dummy_signal = torch.zeros(1, 16000)
        _ = speaker_classifier.encode_batch(dummy_signal)
        print("ECAPA-TDNN model loaded and cached successfully.")
    except Exception as e:
        print(f"Error during ECAPA-TDNN model loading: {e}")
        import traceback
        print(traceback.format_exc())
        print("ECAPA-TDNN model may not have been properly cached.")
        sys.exit(1)

    print("Preloading complete!")


if __name__ == "__main__":
    main()