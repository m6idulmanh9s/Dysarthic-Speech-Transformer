import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from src.training.evaluator import evaluate_batch, compute_wra, compute_cer
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
        
        use_amp = bool(config["training"].get("mixed_precision", True)) and (self.device.type == "cuda")
        self.scaler = GradScaler(device="cuda", enabled=use_amp)
        self.accumulation_steps = config["training"]["accumulation_steps"]
        self.max_grad_norm = config["training"]["max_grad_norm"]

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
       return nn.Transformer.generate_square_subsequent_mask(
            sz, device=self.device, dtype=torch.bool
        )

    def create_padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Returns True for padded positions."""
        batch_size = lengths.size(0)
        lengths = lengths.to(self.device)
        seq_range = torch.arange(max_len, device=self.device).unsqueeze(0).expand(batch_size, max_len)
        return seq_range >= lengths.unsqueeze(1)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0
        start_time = time.time()
        self.optimizer.zero_grad()

        for batch_idx, (specs, tokens, spec_lens, token_lens) in enumerate(self.train_loader):
            specs = specs.to(self.device)
            tokens = tokens.to(self.device)

            tgt_input = tokens[:, :-1]
            tgt_expected = tokens[:, 1:]
            
            tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1))
            src_padding_mask = self.create_padding_mask(spec_lens, specs.size(1))
            tgt_padding_mask = self.create_padding_mask(token_lens - 1, tgt_input.size(1))

            # FIX: Single Forward pass with Automatic Mixed Precision
            with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                logits = self.model(
                    src=specs, 
                    tgt=tgt_input, 
                    tgt_mask=tgt_mask,
                    src_padding_mask=src_padding_mask,
                    tgt_padding_mask=tgt_padding_mask
                )
                output_dim = logits.shape[-1]
                
                logits_flat = logits.contiguous().view(-1, output_dim)
                tgt_expected_flat = tgt_expected.contiguous().view(-1)

                loss = self.criterion(logits_flat, tgt_expected_flat)
                loss_value = loss.item()
                scaled_loss = loss / self.accumulation_steps

            self.scaler.scale(scaled_loss).backward()

            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            epoch_loss += loss_value

            if batch_idx % self.config["wandb"]["log_freq"] == 0:
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {loss_value:.4f}")
                if wandb.run is not None:
                    wandb.log({"Train Batch Loss": loss_value})

        self.scheduler.step()
        
        num_batches = max(len(self.train_loader), 1)
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - start_time
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"=== Epoch {epoch} Complete | Avg Train Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s ===")
        if wandb.run is not None:
            wandb.log({"Epoch Avg Train Loss": avg_loss, "Learning Rate": self.scheduler.get_last_lr()[0]})
            
        return avg_loss

    def validate(self) -> tuple[float, float, float]:
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for specs, tokens, spec_lens, token_lens in self.val_loader:
                specs = specs.to(self.device)
                tokens = tokens.to(self.device)

                tgt_input = tokens[:, :-1]
                tgt_expected = tokens[:, 1:]
                
                tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1))
                src_padding_mask = self.create_padding_mask(spec_lens, specs.size(1))
                tgt_padding_mask = self.create_padding_mask(token_lens - 1, tgt_input.size(1))

                with autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                    logits = self.model(
                        src=specs, 
                        tgt=tgt_input, 
                        tgt_mask=tgt_mask,
                        src_padding_mask=src_padding_mask,
                        tgt_padding_mask=tgt_padding_mask
                    )
                    loss = self.criterion(
                        logits.contiguous().view(-1, logits.shape[-1]), 
                        tgt_expected.contiguous().view(-1)
                    )
                
                val_loss += loss.item()
                preds, tgts = evaluate_batch(logits, tgt_expected, self.tokenizer)
                all_preds.extend(preds)
                all_targets.extend(tgts)

        num_val_batches = max(len(self.val_loader), 1)
        avg_loss = val_loss / num_val_batches
        wra = compute_wra(all_preds, all_targets)
        cer = compute_cer(all_preds, all_targets)
        
        # Log 2 live sample transcriptions to terminal for inspection
        if all_preds:
            logger.info(f"--- Sample Prediction: '{all_preds[0]}'")
            logger.info(f"--- Sample Target:       '{all_targets[0]}'")

        logger.info(f"Validation | Avg Loss: {avg_loss:.4f} | CER: {cer:.2f}% | Exact WRA: {wra:.2f}%")
        if wandb.run is not None:
            wandb.log({"Val Avg Loss": avg_loss, "Val CER": cer, "Val Exact WRA": wra})
        
        return avg_loss, wra, cer