import torch.nn as nn
from src.utils.logger import get_logger

logger = get_logger("FreezeUtils")


def freeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = True


def apply_phase1_freezing(model: nn.Module) -> None:
    """Phase 1: Ensure entire model (Encoder + Decoder) is unfrozen to learn character modeling."""
    logger.info("Applying Phase 1 Setup: Unfreezing full model for healthy speech pretraining.")
    unfreeze_module(model)


def apply_phase2_freezing(model: nn.Module, intelligibility_level: str) -> None:
    """Applies Phase 2 severity-based layer freezing."""
    unfreeze_module(model)

    # Normalize severity string (e.g. "Very Low" -> "very_low", "Medium" -> "mild")
    norm_level = str(intelligibility_level).lower().strip().replace(" ", "_")
    if norm_level == "medium":
        norm_level = "mild"

    if norm_level in ["very_low", "low"]:
        logger.info(f"Applying Phase 2 Freezing ({norm_level}): Freezing last decoder layer.")
        freeze_module(model.decoder.layers[-1])
    elif norm_level in ["mild", "high"]:
        logger.info(f"Applying Phase 2 Freezing ({norm_level}): Freezing last decoder feed-forward layer.")
        freeze_module(model.decoder.layers[-1].linear1)
        freeze_module(model.decoder.layers[-1].linear2)
    else:
        logger.warning(f"Unrecognized severity level '{intelligibility_level}'. Leaving model unfrozen.")