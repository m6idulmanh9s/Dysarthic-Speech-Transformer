from typing import List, Tuple
import torch
from src.utils.tokenizer import CharacterTokenizer


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def compute_cer(predictions: List[str], targets: List[str]) -> float:
    """Computes Character Error Rate (CER) percentage."""
    total_dist = 0
    total_chars = 0
    for pred, tgt in zip(predictions, targets):
        total_dist += levenshtein_distance(pred, tgt)
        total_chars += max(len(tgt), 1)
    return (total_dist / total_chars) * 100.0 if total_chars > 0 else 0.0


def compute_wra(predictions: List[str], targets: List[str]) -> float:
    """Computes exact full-sentence match accuracy."""
    if not targets:
        return 0.0
    correct = sum(1 for pred, tgt in zip(predictions, targets) if pred.strip() == tgt.strip())
    return (correct / len(targets)) * 100.0


def evaluate_batch(logits: torch.Tensor, targets: torch.Tensor, tokenizer: CharacterTokenizer) -> Tuple[List[str], List[str]]:
    """Decodes batch logits and targets into strings."""
    preds = torch.argmax(logits, dim=-1)
    pred_strings = [tokenizer.decode(preds[i].tolist()) for i in range(preds.size(0))]
    tgt_strings = [tokenizer.decode(targets[i].tolist()) for i in range(targets.size(0))]
    return pred_strings, tgt_strings