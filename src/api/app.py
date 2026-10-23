import yaml
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.api.schemas import HealthResponse, TranscriptionResponse
from src.api.inference_engine import SpeechInferenceEngine
from src.utils.logger import get_logger

logger = get_logger("API")

app = FastAPI(
    title="Dysarthric Speech Recognition API",
    description="Sequence-to-Sequence Dysarthric Speech Transformer Inference Endpoint",
    version="1.0.0",
)

# Allow CORS for web-based frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine instance
engine = None


@app.on_event("startup")
async def startup_event():
    global engine
    try:
        with open("configs/model_config.yaml", "r") as f:
            config = yaml.safe_load(f)
            
        # Add audio configs directly for the engine
        with open("configs/data_config.yaml", "r") as f:
            data_config = yaml.safe_load(f)
            config["audio"] = data_config["audio"]

        # Note: Provide the actual path to your trained checkpoint here
        checkpoint_path = "checkpoints/best_dysarthric_model.pt"
        if not checkpoint_path.exists():
            # Fallback to any speaker checkpoint in directory
            ckpt_files = list(Path("checkpoints").glob("best_dysarthric_model_*.pt"))
            if ckpt_files:
                checkpoint_path = ckpt_files[0]
        engine = SpeechInferenceEngine(model_path=checkpoint_path, config=config)
        logger.info("API Startup Complete.")
    except Exception as e:
        logger.error(f"Failed to initialize Inference Engine during startup: {e}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    if engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized.")
        
    return HealthResponse(
        status="active",
        model_loaded=engine.is_loaded,
        device=str(engine.device)
    )


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    if not engine.is_loaded:
        raise HTTPException(status_code=503, detail="Model weights are not loaded.")
        
    if not file.filename.endswith((".wav", ".flac")):
        raise HTTPException(status_code=400, detail="Only .wav or .flac files are supported.")
        
    try:
        audio_bytes = await file.read()
        result = engine.transcribe(audio_bytes)
        
        return TranscriptionResponse(
            transcription=result["transcription"],
            processing_time_sec=result["processing_time_sec"],
            message="Success"
        )
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))