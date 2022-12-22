import os
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

from .scheduler import LinearWarmupLR


class SampleModel(nn.Module):
    """SampleModel for classification

    Attributes:
        input_dimension: input dimension for linear layer
    """

    def __init__(self, input_dimension: int, num_classes: int):
        super().__init__()

        self.feedforward = nn.Linear(input_dimension, num_classes)

    def forward(self, x) -> torch.Tensor:
        """Forward Model

        Args:
            x: input data shaped [BatchSize, InputDimension]
        Returns:
            model output shaped [BatchSize, NumClasses]
        """
        return self.feedforward(x)


class SimpleClassification(pl.LightningModule):
    """Pytorch lightning classification

    Attributes:
        model: model for classification
        num_classes: the number of classes
        total_steps: total training steps for lr scheduling
        max_learning_rate: Max LR
        min_learning_rate: Min LR
        warmup_rate: warmup step rate
        model_save_dir: path to save model
    """

    def __init__(
        self,
        input_dimension: int,
        num_classes: int,
        total_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_rate: float,
        model_save_dir: str,
    ):
        super().__init__()

        self.model = SampleModel(input_dimension=input_dimension, num_classes=num_classes)
        self.num_classes = num_classes

        self.total_steps = total_steps
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_rate = warmup_rate
        self.model_save_dir = model_save_dir

        self.save_hyperparameters(
            {
                "input_dimension": input_dimension,
                "num_classes": num_classes,
                "total_steps": total_steps,
                "max_learning_rate": max_learning_rate,
                "min_learning_rate": min_learning_rate,
                "warmup_rate": warmup_rate,
                "model_save_dir": model_save_dir,
            }
        )

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Train step function

        Args:
            batch: training batch input/label
                batch["input"] shaped [BatchSize, InputDimension]
                batch["label"] shaped [BatchSize]
        Returns:
            metrics dictionary of this train step
        """
        output = self.model(batch["input"])

        loss = F.cross_entropy(output, batch["label"])
        accuracy = torchmetrics.functional.accuracy(
            output, batch["label"], task="multiclass", top_k=1, num_classes=self.num_classes
        )

        metrics = {"loss": loss, "acc": accuracy}
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True)

        return metrics

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step function

        Args:
            batch: validating batch input/label
                batch["input"] shaped [BatchSize, InputDimension]
                batch["label"] shaped [BatchSize]
        Returns:
            metrics dictionary of this validation step
        """
        output = self.model(batch["input"])

        loss = F.cross_entropy(output, batch["label"])
        accuracy = torchmetrics.functional.accuracy(
            output, batch["label"], task="multiclass", top_k=1, num_classes=self.num_classes
        )

        metrics = {"val_loss": loss, "val_acc": accuracy}
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return metrics

    def test_step(self, *args, **kwargs):
        """Same as `validation_step`"""
        return self.validation_step(*args, **kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.max_learning_rate)
        scheduler = LinearWarmupLR(
            optimizer,
            int(self.total_steps * self.warmup_rate),
            self.total_steps,
            self.min_learning_rate / self.max_learning_rate,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "Learning Rate"},
        }

    def validation_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)

        if self.trainer.is_global_zero:
            val_losses = [output["val_loss"].mean() for output in outputs]
            val_accs = [output["val_acc"].mean() for output in outputs]

            val_loss_mean = sum(val_losses) / len(val_losses)
            val_acc_mean = sum(val_accs) / len(val_accs)

            model_save_path = os.path.join(
                self.model_save_dir,
                f"model-{self.current_epoch:02d}epoch-{self.global_step}steps-{val_loss_mean:.4f}loss-{val_acc_mean:.4f}acc",
            )
            torch.save(self.model.state_dict(), model_save_path)
