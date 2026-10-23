from typing import List, Tuple
import torch
from src.utils.tokenizer import CharacterTokenizer


def compute_wra(predictions: List[str], targets: List[str]) -> float:
    """
    Computes Word Recognition Accuracy (WRA).
    WRA is the percentage of words the model correctly identifies[cite: 8].
    """
    if not targets:
        return 0.0

    correct = 0
    for pred, tgt in zip(predictions, targets):
        if pred.strip() == tgt.strip():
            correct += 1

    return (correct / len(targets)) * 100.0


def evaluate_batch(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    tokenizer: CharacterTokenizer
) -> Tuple[List[str], List[str]]:
    """Decodes batch logits and targets into strings for evaluation."""
    # logits shape: (Batch, Seq_Len, Vocab_Size)
    preds = torch.argmax(logits, dim=-1)
    
    pred_strings = []
    tgt_strings = []
    
    for i in range(preds.size(0)):
        pred_str = tokenizer.decode(preds[i].tolist())
        tgt_str = tokenizer.decode(targets[i].tolist())
        pred_strings.append(pred_str)
        tgt_strings.append(tgt_str)
        
    return pred_strings, tgt_strings