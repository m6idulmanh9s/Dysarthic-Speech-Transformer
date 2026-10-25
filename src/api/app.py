from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.inference_engine import InferenceEngine
from src.utils.logger import get_logger

logger = get_logger("API")

BASE_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
MODEL_CONFIG = BASE_DIR / "configs" / "model_config.yaml"
DATA_CONFIG = BASE_DIR / "configs" / "data_config.yaml"
TEMP_DIR = BASE_DIR / "data" / "tmp"

app = FastAPI(
    title="Dysarthric Speech Recognition API",
    description="Sequence-to-Sequence Dysarthric Speech Transformer Inference Endpoint",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None


def resolve_checkpoint() -> Path:
    preferred_checkpoints = [
        CHECKPOINT_DIR / "best_dysarthric_model_F02.pt",
        CHECKPOINT_DIR / "best_dysarthric_model.pt",
        CHECKPOINT_DIR / "phase1_base_model.pt",
    ]

    for checkpoint in preferred_checkpoints:
        if checkpoint.is_file():
            return checkpoint

    available_checkpoints = sorted(CHECKPOINT_DIR.glob("*.pt"))

    if available_checkpoints:
        return available_checkpoints[0]

    raise FileNotFoundError(
        f"No checkpoint files found in {CHECKPOINT_DIR}"
    )


@app.on_event("startup")
def startup_event():
    global engine

    try:
        checkpoint_path = resolve_checkpoint()

        logger.info(
            "Initializing inference engine with checkpoint: %s",
            checkpoint_path,
        )

        engine = InferenceEngine(
            checkpoint_path=checkpoint_path,
            model_config_path=MODEL_CONFIG,
            data_config_path=DATA_CONFIG,
        )

        logger.info(
            "Inference engine initialized successfully on %s",
            engine.device,
        )

    except Exception:
        engine = None
        logger.exception("Failed to initialize inference engine")


@app.get("/health")
def health_check():
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine is not initialized.",
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "device": str(engine.device),
        "checkpoint": str(engine.checkpoint_path),
    }


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine is not ready.",
        )

    filename = file.filename or ""

    if Path(filename).suffix.lower() not in {".wav", ".flac"}:
        raise HTTPException(
            status_code=400,
            detail="Only WAV and FLAC audio files are supported.",
        )

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_file_path = TEMP_DIR / f"{uuid4().hex}{Path(filename).suffix.lower()}"

    try:
        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded audio file is empty.",
            )

        temp_file_path.write_bytes(contents)

        result = engine.transcribe(temp_file_path)

        return {
            "filename": Path(filename).name,
            "transcription": result["transcription"],
            "processing_time_sec": result["processing_time_sec"],
            "message": "Success",
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Transcription failed")
        raise HTTPException(
            status_code=500,
            detail="Transcription failed. Check server logs for details.",
        )

    finally:
        temp_file_path.unlink(missing_ok=True)
        await file.close()