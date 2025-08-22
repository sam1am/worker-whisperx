from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class WordSegment(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None 


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: List[WordSegment] = []
    speaker: Optional[str] = None 


class TranscriberOutput(BaseModel):
    segments: List[Segment] = []
    word_segments: List[WordSegment] = []
    diarize_segments: Optional[Dict[str, Any]] = None
    audio_duration: Optional[int] = None 
    transcription_duration: Optional[int] = None
    alignment_duration: Optional[int] = None
    diarization_duration: Optional[int] = None


INPUT_VALIDATIONS = {
    'audio': {
        'type': str,
        'required': True,
        'default': ''
    },
    'model_name': {
        'type': str,
        'required': False,
        'default': "large-v2"
    },
    'language': {
        'type': str,
        'required': False,
        'default': "en"
    },
    'batch_size': {
        'type': int,
        'required': False,
        'default': 16
    },
    'diarize': {
        'type': bool,
        'required': False,
        'default': False
    },
    'diarization_method': {
        'type': str,
        'required': False,
        'default': "pyannote"  # 'pyannote', 'ecapa_tdnn', 'ahc', or 'speechbrain_verify'
    },
    'min_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    'max_speakers': {
        'type': int,
        'required': False,
        'default': None
    },
    'hf_token': {
        'type': str,
        'required': False,
        'default': None
    },
    'similarity_threshold': {
        'type': float,
        'required': False,
        'default': 0.65
    },
    'distance_threshold': {
        'type': float,
        'required': False,
        'default': 0.5
    }
}