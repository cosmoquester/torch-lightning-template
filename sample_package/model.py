from typing import Dict, Tuple

import lightning as pl
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
        num_classes: the number of classes
        total_steps: total training steps for lr scheduling
        max_learning_rate: Max LR
        min_learning_rate: Min LR
        warmup_rate: warmup step rate
    """

    def __init__(
        self,
        input_dimension: int,
        num_classes: int,
        total_steps: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_rate: float,
    ):
        super().__init__()

        self.model = SampleModel(input_dimension=input_dimension, num_classes=num_classes)
        self.num_classes = num_classes

        self.total_steps = total_steps
        self.max_learning_rate = max_learning_rate
        self.min_learning_rate = min_learning_rate
        self.warmup_rate = warmup_rate

        self.save_hyperparameters(
            {
                "input_dimension": input_dimension,
                "num_classes": num_classes,
                "total_steps": total_steps,
                "max_learning_rate": max_learning_rate,
                "min_learning_rate": min_learning_rate,
                "warmup_rate": warmup_rate,
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

        metrics = {"val/loss": loss, "val/acc": accuracy}
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step function

        Args:
            batch: validating batch input/label
                batch["input"] shaped [BatchSize, InputDimension]
                batch["label"] shaped [BatchSize]
        Returns:
            metrics dictionary of this test step
        """
        output = self.model(batch["input"])

        loss = F.cross_entropy(output, batch["label"])
        accuracy = torchmetrics.functional.accuracy(
            output, batch["label"], task="multiclass", top_k=1, num_classes=self.num_classes
        )

        metrics = {"test/loss": loss, "test/acc": accuracy}
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True)

        return metrics

    def configure_optimizers(
        self, skip_list: Tuple[str] = ("LayerNorm", "layer_norm", "layernorm")
    ) -> Dict[str, object]:
        decay = []
        no_decay = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            for skip_name in skip_list:
                if skip_name in name:
                    no_decay.append(param)
                    break
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            params=[{"params": no_decay, "weight_decay": 0.0}, {"params": decay}], lr=self.max_learning_rate
        )
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
