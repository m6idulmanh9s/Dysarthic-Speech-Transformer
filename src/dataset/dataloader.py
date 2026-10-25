import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from src.dataset.augmentations import AudioAugmenter
from src.utils.tokenizer import CharacterTokenizer


class DysarthricSpeechDataset(Dataset):
    def __init__(
        self,
        csv_file: str,
        tokenizer: CharacterTokenizer,
        sample_rate: int = 16000,
        augment: bool = False,
    ):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.augment = augment
        
        # Fast augmentation helper
        self.augmenter = AudioAugmenter(sample_rate=self.sample_rate) if augment else None
        
        # Spectrogram extraction (STFT)
        self.spectrogram_transform = T.Spectrogram(n_fft=3200, hop_length=160, power=2.0)
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        item = self.df.iloc[idx]

        # Robust audio path lookup
        wav_path = item["audio_filepath"]

        # Robust text transcript lookup
        text_val = item["clean_transcript"]

        # Load audio
        waveform, sr = torchaudio.load(wav_path)
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 1. Waveform Augmentation (Fast Noise Injection)
        if self.augment and self.augmenter:
            waveform = self.augmenter.augment_waveform(waveform)

        # Compute Spectrogram
        spec = self.spectrogram_transform(waveform)
        spec_db = self.amplitude_to_db(spec).squeeze(0)  # [freq_bins, time_steps]

        # 2. Spectrogram Augmentation (SpecAugment Masking)
        if self.augment and self.augmenter:
            spec_db = self.augmenter.apply_spec_augment(spec_db)

        # Transpose to [time_steps, freq_bins]
        spec_db = spec_db.transpose(0, 1)

        # Tokenize label
        tokens = torch.tensor(self.tokenizer.encode(str(text_val)), dtype=torch.long)

        return spec_db, tokens


def collate_fn(batch):
    specs, tokens = zip(*batch)
    
    spec_lens = torch.tensor([s.size(0) for s in specs], dtype=torch.long)
    token_lens = torch.tensor([t.size(0) for t in tokens], dtype=torch.long)

    padded_specs = pad_sequence(specs, batch_first=True, padding_value=0.0)
    padded_tokens = pad_sequence(tokens, batch_first=True, padding_value=0)  # PAD_IDX = 0

    return padded_specs, padded_tokens, spec_lens, token_lens


def create_dataloader(
    csv_file: str,
    tokenizer: CharacterTokenizer,
    batch_size: int = 8,
    shuffle: bool = True,
    augment: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    dataset = DysarthricSpeechDataset(
        csv_file, tokenizer=tokenizer, augment=augment
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,
    )