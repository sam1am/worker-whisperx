<div align="center">

<h1>WhisperX | Worker</h1>  

[WhisperX](https://github.com/m-bain/whisperX?tab=readme-ov-file) for serverless RunPod inferencing. Uses [Faster Whisper](https://github.com/guillaumekln/faster-whisper) as a template. [Docker Image](https://hub.docker.com/r/realyashnag/worker-whisperx) is hosted on Docker Hub.


<h1>Faster Whisperx | Worker</h1>  

This repository contains the [Faster Whisper](https://github.com/guillaumekln/faster-whisper) Worker for RunPod. The Whisper Worker is designed to process audio files using various Whisper models, with options for transcription formatting, language translation, and more. It's part of the RunPod Workers collection aimed at providing diverse functionality for endpoint processing.

[Endpoint Docs](https://docs.runpod.io/reference/faster-whisper)

[Docker Image](https://hub.docker.com/r/runpod/ai-api-faster-whisper)

</div>

## Docker Image

The Docker image comes preloaded with the following models:
- Whisper models: tiny, large-v2
- SpeechBrain ECAPA-TDNN speaker recognition model
- SpeechBrain speaker verification model

For pyannote diarization, you'll need to provide your own Hugging Face token with access to the pyannote models.

### Building with Hugging Face Token

To build the Docker image with a Hugging Face token (for pyannote model preloading):

```bash
docker build --build-arg HF_TOKEN=your_hf_token_here -t worker-whisperx .
```

## Model Inputs

| Input                               | Type  | Description                                                                                                                                              |
|-------------------------------------|-------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `audio`                             | Path  | Audio file                                                                                                                                               |
| `model_name`                        | str   | Choose a Whisper model. Choices: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3". Default: "large-v2"                             |
| `language`                          | str   | Language spoken in the audio, specify None to perform language detection. Default: "en"                                                                  |
| `batch_size`                        | int   | Batch size for transcription. Default: 16                                                                                                                |
| `diarize`                           | bool  | Enable speaker diarization. Default: False                                                                                                               |
| `diarization_method`                | str   | Choose diarization method. Choices: "pyannote", "ecapa_tdnn", "ahc", "speechbrain_verify". Default: "pyannote"                                          |
| `min_speakers`                      | int   | Minimum number of speakers (for pyannote method). Default: None                                                                                          |
| `max_speakers`                      | int   | Maximum number of speakers (for pyannote method). Default: None                                                                                          |
| `hf_token`                          | str   | Hugging Face token for pyannote model access. Default: None                                                                                              |
| `similarity_threshold`              | float | Similarity threshold for ecapa_tdnn method. Default: 0.65                                                                                                |
| `distance_threshold`                | float | Distance threshold for ahc method. Default: 0.5                                                                                                          |

## Diarization

This worker now supports speaker diarization, which identifies and labels different speakers in an audio file. There are several methods available:

### Diarization Methods

1. **Pyannote** (default): Uses the pyannote.audio library for state-of-the-art diarization
   - Most accurate method
   - Requires Hugging Face token with access to pyannote models
   - Supports min_speakers and max_speakers parameters

2. **ECAPA-TDNN**: Uses SpeechBrain's ECAPA-TDNN embeddings with a greedy clustering approach
   - Good balance of accuracy and speed
   - No external dependencies
   - Uses similarity_threshold parameter

3. **AHC**: Uses Agglomerative Hierarchical Clustering with SpeechBrain embeddings
   - Good for when you want to tune clustering behavior
   - No external dependencies
   - Uses distance_threshold parameter

4. **SpeechBrain Verify**: Uses SpeechBrain's speaker verification model
   - Most computationally expensive
   - No external dependencies
   - Uses file-based verification

### Pyannote Diarization (Recommended)

To use pyannote diarization, you'll need a Hugging Face token with access to the pyannote models:

```json
{
    "input": {
        "audio": "https://example.com/audio.mp3",
        "diarize": true,
        "diarization_method": "pyannote",
        "hf_token": "YOUR_HF_TOKEN"
    }
}
```

You can also specify the expected number of speakers:

```json
{
    "input": {
        "audio": "https://example.com/audio.mp3",
        "diarize": true,
        "diarization_method": "pyannote",
        "hf_token": "YOUR_HF_TOKEN",
        "min_speakers": 2,
        "max_speakers": 4
    }
}
```

### Alternative Diarization Methods

For the alternative methods, no Hugging Face token is required:

```json
{
    "input": {
        "audio": "https://example.com/audio.mp3",
        "diarize": true,
        "diarization_method": "ecapa_tdnn"
    }
}
```

Each method has tunable parameters:

```json
{
    "input": {
        "audio": "https://example.com/audio.mp3",
        "diarize": true,
        "diarization_method": "ahc",
        "distance_threshold": 0.4
    }
}
```

```json
{
    "input": {
        "audio": "https://example.com/audio.mp3",
        "diarize": true,
        "diarization_method": "ecapa_tdnn",
        "similarity_threshold": 0.7
    }
}
```

## Performance Metrics

The worker now includes performance metrics in the output to help you understand processing times for each step:

- `audio_duration`: Duration of the input audio in seconds (rounded up)
- `transcription_duration`: Time taken for transcription in seconds (rounded up)
- `alignment_duration`: Time taken for word alignment in seconds (rounded up)
- `diarization_duration`: Time taken for diarization in seconds (rounded up)

These metrics help you understand the processing overhead of each step and optimize your usage accordingly.

## Test Inputs

The following inputs can be used for testing the model:

```json
{
    "input": {
        "audio": "https://github.com/runpod-workers/sample-inputs/raw/main/audio/gettysburg.wav"
    }
}
```

## Sample Output (with Diarization)

```json
{
    "segments": [
        {
            "start": 0.0,
            "end": 3.0,
            "text": "Hello world.",
            "words": [
                {
                    "word": "Hello",
                    "start": 0.0,
                    "end": 1.5,
                    "speaker": "SPEAKER_00"
                },
                {
                    "word": "world.",
                    "start": 1.5,
                    "end": 3.0,
                    "speaker": "SPEAKER_00"
                }
            ],
            "speaker": "SPEAKER_00"
        },
        {
            "start": 3.5,
            "end": 6.0,
            "text": "How are you?",
            "words": [
                {
                    "word": "How",
                    "start": 3.5,
                    "end": 4.0,
                    "speaker": "SPEAKER_01"
                },
                {
                    "word": "are",
                    "start": 4.0,
                    "end": 4.5,
                    "speaker": "SPEAKER_01"
                },
                {
                    "word": "you?",
                    "start": 4.5,
                    "end": 6.0,
                    "speaker": "SPEAKER_01"
                }
            ],
            "speaker": "SPEAKER_01"
        }
    ],
    "word_segments": [
        {
            "word": "Hello",
            "start": 0.0,
            "end": 1.5,
            "speaker": "SPEAKER_00"
        },
        {
            "word": "world.",
            "start": 1.5,
            "end": 3.0,
            "speaker": "SPEAKER_00"
        },
        {
            "word": "How",
            "start": 3.5,
            "end": 4.0,
            "speaker": "SPEAKER_01"
        },
        {
            "word": "are",
            "start": 4.0,
            "end": 4.5,
            "speaker": "SPEAKER_01"
        },
        {
            "word": "you?",
            "start": 4.5,
            "end": 6.0,
            "speaker": "SPEAKER_01"
        }
    ],
    "audio_duration": 6,
    "transcription_duration": 2,
    "alignment_duration": 1,
    "diarization_duration": 3
}
```
