from typing import Dict, List


class CharacterTokenizer:
    """Character-level tokenizer mapping characters to integer indices."""

    def __init__(self):
        self.chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '"

        self.PAD_IDX = 0
        self.SOS_IDX = 1
        self.EOS_IDX = 2
        self.UNK_IDX = 3

        self.char2idx: Dict[str, int] = {
            "<PAD>": self.PAD_IDX,
            "<SOS>": self.SOS_IDX,
            "<EOS>": self.EOS_IDX,
            "<UNK>": self.UNK_IDX,
        }

        for idx, char in enumerate(self.chars):
            self.char2idx[char] = idx + 4

        self.idx2char: Dict[int, str] = {
            idx: char for char, idx in self.char2idx.items()
        }
        self.vocab_size = len(self.char2idx)

    def encode(self, text: str) -> List[int]:
        tokens = [self.SOS_IDX]
        for char in text:
            tokens.append(self.char2idx.get(char, self.UNK_IDX))
        tokens.append(self.EOS_IDX)
        return tokens

    def decode(self, tokens: List[int]) -> str:
        chars = []
        for token in tokens:
            if token in (self.PAD_IDX, self.SOS_IDX, self.EOS_IDX):
                continue
            chars.append(self.idx2char.get(token, ""))
        return "".join(chars)