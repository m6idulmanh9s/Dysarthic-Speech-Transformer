import pytest
from src.utils.tokenizer import CharacterTokenizer


def test_tokenizer_encoding_decoding():
    tokenizer = CharacterTokenizer()
    text = "HELLO WORLD 123"

    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    assert encoded[0] == tokenizer.SOS_IDX
    assert encoded[-1] == tokenizer.EOS_IDX
    assert decoded == text


def test_tokenizer_unknown_character():
    tokenizer = CharacterTokenizer()
    text = "HELLO @ WORLD"

    encoded = tokenizer.encode(text)
    assert tokenizer.UNK_IDX in encoded