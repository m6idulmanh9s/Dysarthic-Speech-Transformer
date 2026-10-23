from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class TranscriptionResponse(BaseModel):
    transcription: str
    processing_time_sec: float
    message: str = "Success"