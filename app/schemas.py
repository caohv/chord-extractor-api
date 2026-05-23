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


class MeterResponse(BaseModel):
    duration: float
    beats_per_bar: int
    time_signature: str
    confidence: float
    downbeats: list[float]


class Section(BaseModel):
    start: float
    end: float
    label: str


class LyricLine(BaseModel):
    start: float
    end: float
    text: str
    label: str  # structural label from segments[]; "unknown" if no segment covers


class SectionsResponse(BaseModel):
    duration: float
    bpm: int
    beats: list[float]
    downbeats: list[float]
    segments: list[Section]
    lyrics: list[LyricLine] | None = None
