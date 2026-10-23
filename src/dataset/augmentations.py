import torch
import torchaudio.transforms as T


class AudioAugmenter:
    """Applies on-the-fly waveform perturbations for data augmentation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        pitch_shift_steps: int = 2,
        noise_scale: float = 0.005,
    ):
        self.sample_rate = sample_rate
        self.pitch_shift_steps = pitch_shift_steps
        self.noise_scale = noise_scale
        self.pitch_shifter = T.PitchShift(self.sample_rate, n_steps=self.pitch_shift_steps)

    def apply_pitch_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        
        return pitch_shifter(waveform)

    def apply_noise_injection(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(waveform) * self.noise_scale
        return waveform + noise

    def augment(self, waveform: torch.Tensor) -> torch.Tensor:
        choice = torch.randint(0, 2, (1,)).item()
        if choice == 0:
            return self.apply_pitch_shift(waveform)
        else:
            return self.apply_noise_injection(waveform)