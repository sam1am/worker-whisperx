#!/usr/bin/env python3
"""
Setup required models - ensures they're in the locations where they'll be searched for
"""
import os
import sys
import torch
import argparse
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define cache directories
WHISPERX_CACHE = "/tmp"  # Where predict.py looks for WhisperX models
TORCH_CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints")
HF_CACHE = os.path.expanduser("~/.cache/huggingface")


def ensure_dirs():
    """Ensure all cache directories exist"""
    os.makedirs(WHISPERX_CACHE, exist_ok=True)
    os.makedirs(TORCH_CACHE, exist_ok=True)
    os.makedirs(HF_CACHE, exist_ok=True)
    # Also ensure the parent directories exist
    os.makedirs(os.path.dirname(TORCH_CACHE), exist_ok=True)
    os.makedirs(os.path.dirname(HF_CACHE), exist_ok=True)


def download_wav2vec2():
    """Download the wav2vec2 model directly to torch cache"""
    model_file = "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
    model_path = os.path.join(TORCH_CACHE, model_file)

    if os.path.exists(model_path):
        print(f"wav2vec2 model already exists at {model_path}")
        return

    print(f"Downloading wav2vec2 model to {model_path}...")
    try:
        import torchaudio
        torchaudio.utils.download_asset(
            "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth",
            model_path
        )
        print(f"Downloaded wav2vec2 model to {model_path}")
    except Exception as e:
        print(f"Error downloading wav2vec2 model: {e}")


def download_huggingface_models(hf_token):
    """Download the segmentation and diarization models from HuggingFace directly to cache"""
    if not hf_token:
        print("Skipping HuggingFace models download - no token provided")
        return

    # Setting the token in environment
    os.environ["HF_TOKEN"] = hf_token

    try:
        print("Downloading HuggingFace models to default cache location...")
        from pyannote.audio import Model, Pipeline

        # Configure HuggingFace to use our cache location
        import huggingface_hub
        huggingface_hub.constants.HF_CACHE = HF_CACHE

        # Download segmentation model
        print("Downloading segmentation model...")
        Model.from_pretrained("pyannote/segmentation-3.0",
                              use_auth_token=hf_token)

        # Download diarization model
        print("Downloading speaker diarization model...")
        Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=hf_token)

        print("Successfully downloaded HuggingFace models")
    except Exception as e:
        print(f"Error downloading HuggingFace models: {e}")
        import traceback
        traceback.print_exc()


def download_whisperx_models():
    """Download the whisperx models directly to /tmp where predict.py looks for them"""
    whisperx_models = ["tiny", "large-v2"]

    try:
        print(f"Downloading WhisperX models to {WHISPERX_CACHE}...")
        import whisperx

        # Pre-download tiny and large-v2 models that are used in the predictor
        for model in whisperx_models:
            model_path = os.path.join(WHISPERX_CACHE, model)
            if os.path.exists(model_path):
                print(
                    f"WhisperX model '{model}' already exists at {model_path}")
            else:
                print(
                    f"Downloading WhisperX model '{model}' to {WHISPERX_CACHE}")
                whisperx.load_model(
                    model, compute_type="float16", download_root=WHISPERX_CACHE)

        print("Successfully downloaded WhisperX models")
    except Exception as e:
        print(f"Error downloading WhisperX models: {e}")
        import traceback
        traceback.print_exc()


def print_model_locations():
    """Print where the models are expected to be"""
    print("\nModel locations:")
    print(f"WhisperX models: {WHISPERX_CACHE}")
    print(f"Wav2Vec2 model: {TORCH_CACHE}")
    print(f"HuggingFace models: {HF_CACHE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Setup models in their correct locations")
    parser.add_argument("--hf_token", type=str, help="HuggingFace access token",
                        default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    # Create necessary directories
    ensure_dirs()

    # Display where models will be downloaded
    print_model_locations()

    # Attempt to download all models to correct locations
    download_wav2vec2()
    download_whisperx_models()

    # Only download HF models if token is provided
    if args.hf_token:
        download_huggingface_models(args.hf_token)
    else:
        print("\nNo HuggingFace token provided. Skipping HF model download.")
        print("Please provide a token with --hf_token or set the HF_TOKEN environment variable")

    print("\nModel setup completed!")
