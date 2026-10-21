import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from src.dataset.augmentations import AudioAugmenter
from src.utils.tokenizer import CharacterTokenizer


class DysarthricSpeechDataset(Dataset):
    """PyTorch Dataset generating voicegrams on-the-fly without static disk overhead."""

    def __init__(
        self,
        csv_file: str,
        tokenizer: CharacterTokenizer,
        target_sr: int = 16000,
        win_length_ms: float = 200.0,
        hop_length_ms: float = 80.0,
        augment: bool = False,
    ):
        self.df = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.target_sr = target_sr
        self.augment = augment

        win_length = int(target_sr * (win_length_ms / 1000.0))  # 3200
        hop_length = int(target_sr * (hop_length_ms / 1000.0))  # 1280
        n_fft = max(256, win_length)

        self.spectrogram_transform = T.Spectrogram(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=1.0
        )
        self.db_transform = T.AmplitudeToDB(top_db=80.0)
        self.augmenter = AudioAugmenter(sample_rate=target_sr) if augment else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        audio_path = row["audio_filepath"]

        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=self.target_sr)
            waveform = resampler(waveform)

        if self.augment and self.augmenter is not None:
            waveform = self.augmenter.augment(waveform)

        # Compute voicegram: (Freq_Bins, Time) -> (Time, Freq_Bins)
        spec = self.spectrogram_transform(waveform)
        spec_db = self.db_transform(spec).squeeze(0).T

        tokens = torch.LongTensor(
            self.tokenizer.encode(str(row["clean_transcript"]))
        )
        return spec_db, tokens


def collate_fn(batch):
    specs, tokens = zip(*batch)

    spec_lengths = torch.LongTensor([s.size(0) for s in specs])
    token_lengths = torch.LongTensor([t.size(0) for t in tokens])

    specs_padded = pad_sequence(specs, batch_first=True, padding_value=0.0)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)

    return specs_padded, tokens_padded, spec_lengths, token_lengths


def create_dataloader(
    csv_file: str,
    tokenizer: CharacterTokenizer,
    batch_size: int = 16,
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
        pin_memory=True,
    )