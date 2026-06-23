from pydantic import BaseModel, HttpUrl


class ExtractRequest(BaseModel):
    url: HttpUrl
    # Optional canonical lyrics (one entry per line/phrase as it appears in
    # the song, with repetitions expanded). When present, /sections uses
    # Whisper purely as a timing source and force-aligns these lines against
    # the audio via Needleman-Wunsch on tokens. When absent, no transcription
    # runs.
    lyrics: list[str] | None = None


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
    # `start`/`end` are None when the line couldn't be aligned to any audio
    # (e.g. a backing-vocal line Whisper missed). In that case `label` is
    # "unaligned"; otherwise it's the structural section ("verse", "chorus",
    # ...) whose interval contains the line's midpoint.
    start: float | None
    end: float | None
    text: str
    label: str


class SectionsResponse(BaseModel):
    duration: float
    bpm: int
    beats: list[float]
    downbeats: list[float]
    segments: list[Section]
    lyrics: list[LyricLine] | None = None
