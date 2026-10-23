import time
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from src.training.evaluator import evaluate_batch, compute_wra
from src.utils.logger import get_logger
from src.utils.tokenizer import CharacterTokenizer

logger = get_logger("Trainer")


class Seq2SeqTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: CharacterTokenizer,
        config: dict,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config["training"]["device"])
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.PAD_IDX)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["epochs"],
            eta_min=float(config["training"]["min_learning_rate"]),
        )
        self.scaler = GradScaler()
        self.accumulation_steps = config["training"]["accumulation_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask.to(self.device)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0
        start_time = time.time()
        self.optimizer.zero_grad()

        if batch_idx % self.config["wandb"]["log_freq"] == 0:
            logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item() * self.accumulation_steps:.4f}")
            if wandb.run is not None:
                wandb.log({"Train Batch Loss": loss.item() * self.accumulation_steps})

        for batch_idx, (specs, tokens, spec_lens, token_lens) in enumerate(self.train_loader):
            specs = specs.to(self.device)
            tokens = tokens.to(self.device)

            tgt_input = tokens[:, :-1]
            tgt_expected = tokens[:, 1:]
            tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1))

            with autocast(device_type=self.device.type, enabled=self.config["training"]["mixed_precision"]):
                logits = self.model(specs, tgt_input, tgt_mask=tgt_mask)
                output_dim = logits.shape[-1]
                logits = logits.contiguous().view(-1, output_dim)
                tgt_expected = tgt_expected.contiguous().view(-1)

                loss = self.criterion(logits, tgt_expected)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                # Tightened gradient clipping to prevent NaN explosion[cite: 1]
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss.item() * self.accumulation_steps

            if batch_idx % self.config["wandb"]["log_freq"] == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss.item() * self.accumulation_steps:.4f}")
                wandb.log({"Train Batch Loss": loss.item() * self.accumulation_steps})

        self.scheduler.step()
        
        avg_loss = epoch_loss / len(self.train_loader)
        epoch_time = time.time() - start_time
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"=== Epoch {epoch} Complete | Avg Train Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s ===")
        wandb.log({"Epoch Avg Train Loss": avg_loss, "Learning Rate": self.scheduler.get_last_lr()[0]})
        if wandb.run is not None:
            wandb.log({"Epoch Avg Train Loss": avg_loss, "Learning Rate": self.scheduler.get_last_lr()[0]})
        
        return avg_loss

    def validate(self) -> float:
        self.model.eval()
        val_loss = 0
        all_preds = []
        all_targets = []

        if wandb.run is not None:
            wandb.log({"Val Avg Loss": avg_loss, "Val WRA": wra})

        with torch.no_grad():
            for specs, tokens, _, _ in self.val_loader:
                specs = specs.to(self.device)
                tokens = tokens.to(self.device)

                tgt_input = tokens[:, :-1]
                tgt_expected = tokens[:, 1:]
                tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1))

                with autocast(device_type=self.device.type, enabled=self.config["training"]["mixed_precision"]):
                    logits = self.model(specs, tgt_input, tgt_mask=tgt_mask)
                    loss = self.criterion(
                        logits.contiguous().view(-1, logits.shape[-1]), 
                        tgt_expected.contiguous().view(-1)
                    )
                
                val_loss += loss.item()
                preds, tgts = evaluate_batch(logits, tgt_expected, self.tokenizer)
                all_preds.extend(preds)
                all_targets.extend(tgts)

        avg_loss = val_loss / len(self.val_loader)
        wra = compute_wra(all_preds, all_targets)
        
        logger.info(f"Validation | Avg Loss: {avg_loss:.4f} | WRA: {wra:.2f}%")
        wandb.log({"Val Avg Loss": avg_loss, "Val WRA": wra})
        
        return wra