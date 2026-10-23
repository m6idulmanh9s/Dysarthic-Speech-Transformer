import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger("FreezeUtils")


def freeze_module(module: nn.Module) -> None:
    """Sets requires_grad to False for all parameters in a module[cite: 7]."""
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    """Sets requires_grad to True for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def apply_phase1_freezing(model: nn.Module) -> None:
    """
    Applies neural freezing for Phase 1 transfer learning (Control Models).
    For Transformer-2, all three decoders are frozen[cite: 7].
    """
    logger.info("Applying Phase 1 Freezing: Freezing all decoder layers.")
    freeze_module(model.decoder)


def apply_phase2_freezing(model: nn.Module, intelligibility_level: str) -> None:
    """
    Applies neural freezing for Phase 2 transfer learning (Dysarthric Models) 
    based on the subject's intelligibility level[cite: 8].
    """
    # Unfreeze everything first to reset state
    unfreeze_module(model)

    if intelligibility_level in ["very_low", "low"]:
        logger.info("Applying Phase 2 Freezing (Very Low/Low): Freezing last decoder[cite: 8].")
        freeze_module(model.decoder.layers[-1])
    
    elif intelligibility_level in ["mild", "high"]:
        logger.info(
            "Applying Phase 2 Freezing (Mild/High): Freezing last decoder's feed-forward component[cite: 8]."
        )
        # In PyTorch's native TransformerDecoderLayer, the FFN consists of linear1 and linear2
        freeze_module(model.decoder.layers[-1].linear1)
        freeze_module(model.decoder.layers[-1].linear2)
    else:
        logger.warning(f"Unknown intelligibility level: {intelligibility_level}. No layers frozen.")