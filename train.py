import argparse
import pandas as pd
import yaml
import torch
from pathlib import Path
import wandb

from src.dataset.dataloader import create_dataloader
from src.models.transformer import DysarthricSpeechTransformer
from src.training.freeze_utils import apply_phase1_freezing, apply_phase2_freezing
from src.training.trainer import Seq2SeqTrainer
from src.utils.tokenizer import CharacterTokenizer
from src.utils.logger import get_logger

logger = get_logger("TrainCLI")

def load_configs(config_dir: str = "configs"):
    config = {}
    for cfg_name in ["data_config.yaml", "model_config.yaml", "train_config.yaml"]:
        with open(f"{config_dir}/{cfg_name}", "r") as f:
            config.update(yaml.safe_load(f))
    return config

def prepare_filtered_csv(df: pd.DataFrame, output_path: str) -> str:
    """Writes a filtered dataframe to a temporary CSV for the dataloader."""
    df.to_csv(output_path, index=False)
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Dysarthric Speech Transformer Training CLI")
    parser.add_argument("--phase", type=int, required=True, choices=[1, 2], help="1 for Base Pre-training, 2 for Speaker Fine-Tuning")
    parser.add_argument("--speaker", type=str, help="Target speaker ID for Phase 2 (e.g., F02)")
    args = parser.parse_args()

    config = load_configs()
    tokenizer = CharacterTokenizer()
    
    # Initialize Model
    model = DysarthricSpeechTransformer(
        input_freq_bins=config["model"]["input_freq_bins"],
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        num_encoder_layers=config["model"]["num_encoder_layers"],
        num_decoder_layers=config["model"]["num_decoder_layers"]
    )

    df_master = pd.read_csv(config["dataset"]["master_csv"])
    Path("data/tmp").mkdir(parents=True, exist_ok=True)

    if args.phase == 1:
        logger.info("Starting Phase 1: Pre-training on Healthy Speech[cite: 7]")
        
        # Filter for healthy speech
        df_healthy = df_master[df_master["is_dysarthric"] == False]
        
        # 90/10 Train/Val Split for Phase 1
        train_df = df_healthy.sample(frac=0.9, random_state=42)
        val_df = df_healthy.drop(train_df.index)
        
        train_csv = prepare_filtered_csv(train_df, "data/tmp/phase1_train.csv")
        val_csv = prepare_filtered_csv(val_df, "data/tmp/phase1_val.csv")
        
        apply_phase1_freezing(model)
        checkpoint_out = "checkpoints/phase1_base_model.pt"

    elif args.phase == 2:
        if not args.speaker:
            raise ValueError("Phase 2 requires a --speaker argument.")
            
        logger.info(f"Starting Phase 2: Fine-Tuning for speaker {args.speaker}[cite: 8]")
        
        # Load Phase 1 weights
        base_ckpt = "checkpoints/phase1_base_model.pt"
        if not Path(base_ckpt).exists():
            raise FileNotFoundError(f"Phase 1 checkpoint missing at {base_ckpt}. Run --phase 1 first.")
        
        model.load_state_dict(torch.load(base_ckpt))
        
        # Filter for target speaker
        df_speaker = df_master[df_master["speaker_id"] == args.speaker]
        if df_speaker.empty:
            raise ValueError(f"No records found for speaker {args.speaker}.")
            
        severity = df_speaker.iloc[0]["severity"]
        
        # Train/Test block split based on updated metadata
        train_df = df_speaker[df_speaker["block"] == "train"]
        val_df = df_speaker[df_speaker["block"] == "test"]

        if train_df.empty or val_df.empty:
            logger.info(f"Speaker {args.speaker} missing dedicated train/test split. Creating 80/20 within-speaker split.")
            train_df = df_speaker.sample(frac=0.8, random_state=42)
            val_df = df_speaker.drop(train_df.index)
        
        train_csv = prepare_filtered_csv(train_df, f"data/tmp/phase2_{args.speaker}_train.csv")
        val_csv = prepare_filtered_csv(val_df, f"data/tmp/phase2_{args.speaker}_val.csv")
        
        apply_phase2_freezing(model, intelligibility_level=severity)
        checkpoint_out = f"checkpoints/best_dysarthric_model_{args.speaker}.pt"

    # Create Dataloaders
    batch_size = config["training"]["batch_size"]
    train_loader = create_dataloader(train_csv, tokenizer, batch_size=batch_size, augment=config["augmentation"]["enabled"])
    val_loader = create_dataloader(val_csv, tokenizer, batch_size=batch_size, shuffle=False)

    # Initialize W&B run
    run_name = f"phase1_healthy_pretrain" if args.phase == 1 else f"phase2_{args.speaker}"
    wandb.init(
        project=config["wandb"]["project"],
        name=run_name,
        config=config,
        # Set mode to "offline" or "disabled" if you do not want to log online
        mode="online"  
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(model, train_loader, val_loader, tokenizer, config)
    
    # Train Loop
    epochs = config["training"]["epochs"]
    best_val_loss = float("inf")
    
    for epoch in range(1, epochs + 1):
        trainer.train_epoch(epoch)
        val_loss, wra, cer = trainer.validate()
        
        # Save model whenever validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_out)
            logger.info(f"--> Saved best model checkpoint to {checkpoint_out} (Val Loss: {best_val_loss:.4f} | CER: {cer:.2f}%)")
    wandb.finish()

if __name__ == "__main__":
    main()