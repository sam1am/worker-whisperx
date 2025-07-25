from typing import Optional, List, Dict, Any
from pydantic import BaseModel


class WordSegment(BaseModel):
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    speaker: Optional[str] = None  # Add speaker field


class Segment(BaseModel):
    start: float
    end: float
    text: str
    words: List[WordSegment] = []
    speaker: Optional[str] = None  # Add speaker field


class TranscriberOutput(BaseModel):
    segments: List[Segment] = []
    word_segments: List[WordSegment] = []
    diarize_segments: Optional[Dict[str, Any]
                               ] = None  # Add diarization segments


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
        'default': None
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
        'default': "pyannote"  # 'pyannote' or 'ecapa_tdnn'
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
    }
}