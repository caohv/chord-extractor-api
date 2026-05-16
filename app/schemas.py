from pydantic import BaseModel, HttpUrl


class ExtractRequest(BaseModel):
    url: HttpUrl


class Chord(BaseModel):
    chord: str
    timestamp: float


class ExtractResponse(BaseModel):
    duration: float
    bpm: float
    bpm_raw: float
    chords: list[Chord]


class BpmResponse(BaseModel):
    duration: float
    bpm: float
    bpm_raw: float


class MeterResponse(BaseModel):
    duration: float
    beats_per_bar: int
    time_signature: str
    confidence: float
    downbeats: list[float]
