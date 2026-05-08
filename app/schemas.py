from pydantic import BaseModel, HttpUrl


class ExtractRequest(BaseModel):
    url: HttpUrl


class Chord(BaseModel):
    chord: str
    timestamp: float


class ExtractResponse(BaseModel):
    duration: float
    bpm: float
    chords: list[Chord]


class BpmResponse(BaseModel):
    duration: float
    bpm: float
