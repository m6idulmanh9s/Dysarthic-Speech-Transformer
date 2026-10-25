import time
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import yaml
from torch.amp import autocast

from src.models.transformer import DysarthricSpeechTransformer
from src.utils.tokenizer import CharacterTokenizer
from src.utils.logger import get_logger

logger = get_logger("InferenceEngine")


class InferenceEngine:
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        model_config_path: Union[str, Path] = "configs/model_config.yaml",
        data_config_path: Union[str, Path] = "configs/data_config.yaml",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_config_path = Path(model_config_path)
        self.data_config_path = Path(data_config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = CharacterTokenizer()

        for path in (
            self.checkpoint_path,
            self.model_config_path,
            self.data_config_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(f"Required file not found: {path}")

        with self.model_config_path.open("r", encoding="utf-8") as file:
            model_config = yaml.safe_load(file)["model"]

        with self.data_config_path.open("r", encoding="utf-8") as file:
            data_config = yaml.safe_load(file)["audio"]

        self.sample_rate = int(data_config["sample_rate"])
        self.input_freq_bins = int(model_config["input_freq_bins"])

        self.model = DysarthricSpeechTransformer(
            input_freq_bins=model_config["input_freq_bins"],
            vocab_size=self.tokenizer.vocab_size,
            d_model=model_config["d_model"],
            n_heads=model_config["n_heads"],
            num_encoder_layers=model_config["num_encoder_layers"],
            num_decoder_layers=model_config["num_decoder_layers"],
        ).to(self.device)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.spectrogram_transform = T.Spectrogram(
            n_fft=int(data_config["n_fft"]),
            win_length=int(data_config["win_length"]),
            hop_length=int(data_config["hop_length"]),
            power=2.0,
        ).to(self.device)

        self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0).to(self.device)

        logger.info("Model loaded successfully on %s", self.device)

    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        return nn.Transformer.generate_square_subsequent_mask(
            size,
            device=self.device,
        )

    def preprocess_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Downmix stereo multi-channel audio to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to target sample rate
        if sample_rate != self.sample_rate:
            waveform = T.Resample(
                orig_freq=sample_rate,
                new_freq=self.sample_rate,
            )(waveform)

        waveform = waveform.to(self.device)

        # Compute Spectrogram & Convert to dB: [1, freq_bins, time_steps]
        spec = self.spectrogram_transform(waveform)
        spec_db = self.amplitude_to_db(spec)

        # Transpose to [1, time_steps, freq_bins]
        spec_db = spec_db.squeeze(0).transpose(0, 1).unsqueeze(0)

        if spec_db.size(-1) != self.input_freq_bins:
            logger.warning(
                "Spectrogram frequency bins (%d) mismatch expected input_freq_bins (%d).",
                spec_db.size(-1),
                self.input_freq_bins,
            )

        return spec_db

    @torch.inference_mode()
    def transcribe(
        self,
        audio_path: Union[str, Path],
        max_len: int = 150,
    ) -> dict:
        start_time = time.perf_counter()
        src = self.preprocess_audio(audio_path)

        generated_tokens = [self.tokenizer.SOS_IDX]
        use_amp = self.device.type == "cuda"

        for _ in range(max_len):
            tgt = torch.tensor(
                [generated_tokens],
                dtype=torch.long,
                device=self.device,
            )
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1))

            with autocast(
                device_type=self.device.type,
                enabled=use_amp,
            ):
                logits = self.model(
                    src=src,
                    tgt=tgt,
                    tgt_mask=tgt_mask,
                )

            next_token = int(logits[0, -1, :].argmax().item())

            if next_token == self.tokenizer.EOS_IDX:
                break

            generated_tokens.append(next_token)

        special_tokens = {
            self.tokenizer.SOS_IDX,
            self.tokenizer.EOS_IDX,
            self.tokenizer.PAD_IDX,
        }

        filtered_tokens = [
            token for token in generated_tokens
            if token not in special_tokens
        ]

        transcription = self.tokenizer.decode(filtered_tokens).strip()
        processing_time = time.perf_counter() - start_time

        return {
            "transcription": transcription,
            "processing_time_sec": round(processing_time, 3),
        }


SpeechInferenceEngine = InferenceEngine