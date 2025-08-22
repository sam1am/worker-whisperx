from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
import os

model_names = ["tiny", "large-v2"]
model_dir = "/tmp"


def load_model(selected_model):
    '''
    Load and cache models in parallel
    '''
    for _attempt in range(5):
        try:
            loaded_model = WhisperModel(
                selected_model,
                device="cpu",
                compute_type="int8",
                download_root=model_dir  # Add this line
            )
            break
        except (AttributeError, OSError):
            if _attempt == 4:  # Last attempt
                raise
            continue

    return selected_model, loaded_model


models = {}

# Ensure model directory exists
os.makedirs(model_dir, exist_ok=True)

with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_model, model_names):
        if model_name is not None:
            models[model_name] = model

print(f"Successfully cached models: {list(models.keys())}")
