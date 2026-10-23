import time
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from src.models.transformer import DysarthricSpeechTransformer
from src.utils.tokenizer import CharacterTokenizer
from src.utils.logger import get_logger

logger = get_logger("InferenceEngine")


class SpeechInferenceEngine:
    """Handles audio preprocessing and autoregressive decoding for the Seq2Seq Transformer."""

    def __init__(self, model_path: str, config: dict):
        self.config = config
        self.device = torch.device("cpu")  # Default to CPU for free-tier web serving
        self.tokenizer = CharacterTokenizer()
        
        # Initialize model architecture based on config
        self.model = DysarthricSpeechTransformer(
            input_freq_bins=config["model"]["input_freq_bins"],
            vocab_size=self.tokenizer.vocab_size,
            d_model=config["model"]["d_model"],
            n_heads=config["model"]["n_heads"],
            num_encoder_layers=config["model"]["num_encoder_layers"],
            num_decoder_layers=config["model"]["num_decoder_layers"],
        ).to(self.device)

        # Load weights safely for CPU inference
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.is_loaded = True
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            self.is_loaded = False

        # Audio transforms matching the dataloader[cite: 1]
        target_sr = config["audio"]["sample_rate"]
        win_length = int(target_sr * (config["audio"]["frame_length_ms"] / 1000.0))
        hop_length = int(target_sr * (config["audio"]["stride_ms"] / 1000.0))
        
        self.spectrogram_transform = T.Spectrogram(
            n_fft=config["audio"]["n_fft"], 
            win_length=win_length, 
            hop_length=hop_length, 
            power=1.0
        ).to(self.device)
        self.db_transform = T.AmplitudeToDB(top_db=80.0).to(self.device)

    def preprocess_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """Converts raw audio bytes into a normalized dB spectrogram tensor."""
        import io
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        target_sr = self.config["audio"]["sample_rate"]
        if sr != target_sr:
            waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
            
        waveform = waveform.to(self.device)
        spec = self.spectrogram_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0).T  # Shape: (Time, Freq_Bins)
        
        return spec_db.unsqueeze(0)  # Add batch dimension: (1, Time, Freq_Bins)

    @torch.no_grad()
    def transcribe(self, audio_bytes: bytes, max_len: int = 150) -> dict:
        """Autoregressive greedy decoding for sequence-to-sequence generation."""
        start_time = time.time()
        
        src = self.preprocess_audio(audio_bytes)
        
        # 1. Encode Acoustic Features
        memory = self.model.frontend(src)
        memory = self.model.pos_encoder(memory)
        for encoder in self.model.encoders:
            memory = encoder(memory)
            
        # 2. Decode Text (Autoregressive loop)
        tgt_tokens = [self.tokenizer.SOS_IDX]
        
        for _ in range(max_len):
            tgt_tensor = torch.LongTensor([tgt_tokens]).to(self.device)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(1)).to(self.device)
            
            # Embed target and decode
            import math
            tgt_embed = self.model.tgt_emb(tgt_tensor) * math.sqrt(self.model.d_model)
            tgt_embed = self.model.pos_decoder(tgt_embed)
            
            output = self.model.decoder(
                tgt=tgt_embed,
                memory=memory,
                tgt_mask=tgt_mask
            )
            
            logits = self.model.out(output)
            next_token = logits[0, -1, :].argmax().item()
            
            if next_token == self.tokenizer.EOS_IDX:
                break
                
            tgt_tokens.append(next_token)
            
        transcription = self.tokenizer.decode(tgt_tokens)
        processing_time = time.time() - start_time
        
        return {
            "transcription": transcription,
            "processing_time_sec": round(processing_time, 3)
        }