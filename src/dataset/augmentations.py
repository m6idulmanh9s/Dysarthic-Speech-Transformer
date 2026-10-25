import torch
import torchaudio.transforms as T

class AudioAugmenter:
    """Fast on-the-fly waveform and spectrogram perturbations."""

    def __init__(
        self,
        sample_rate: int = 16000,
        noise_scale: float = 0.005,
        freq_mask_param: int = 15,
        time_mask_param: int = 35,
    ):
        self.sample_rate = sample_rate
        self.noise_scale = noise_scale
        # SpecAugment transforms (fast tensor masking)
        self.freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_mask = T.TimeMasking(time_mask_param=time_mask_param)

    def apply_noise_injection(self, waveform: torch.Tensor) -> torch.Tensor:
        """Adds Gaussian thermal noise to waveform."""
        noise = torch.randn_like(waveform) * self.noise_scale
        return waveform + noise

    def apply_spec_augment(self, spec_db: torch.Tensor) -> torch.Tensor:
        """Applies frequency and time masking directly to spectrogram tensor."""
        spec_masked = self.freq_mask(spec_db)
        spec_masked = self.time_mask(spec_masked)
        return spec_masked

    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Fast waveform augmentation (Noise injection)."""
        return self.apply_noise_injection(waveform)