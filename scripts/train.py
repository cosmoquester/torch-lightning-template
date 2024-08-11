import argparse
import os

import lightning as pl
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader

from sample_package.data import SampleDataset
from sample_package.model import SimpleClassification
from sample_package.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Train Dialogue Summarization with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, required=True, help="output directory path to save artifacts")
g.add_argument("--train-dataset-pattern", type=str, help="glob pattern of train dataset files")
g.add_argument("--valid-dataset-pattern", type=str, help="glob pattern of valid dataset files")
g.add_argument("--input-dimension", type=int, default=128, help="model input dimension")
g.add_argument("--num-classes", type=int, default=2, help="model input dimension")
g.add_argument("--pretrained-model-path", type=str, help="pretrained model path")
g.add_argument("--batch-size", type=int, default=128, help="training batch size per device")
g.add_argument("--valid-batch-size", type=int, default=256, help="validation batch size per device")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help="the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=10, help="the number of training epochs")
g.add_argument("--max-learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--min-learning-rate", type=float, default=1e-5, help="min Learning rate")
g.add_argument("--warmup-rate", type=float, default=0.05, help="warmup step rate")
g.add_argument("--logging-interval", type=int, default=10, help="logging interval")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--gpus", type=int, help="the number of gpus, use all devices by default")

g = parser.add_argument_group("Wandb Options")
g.add_argument("--wandb-run-name", type=str, help="wanDB run name")
g.add_argument("--wandb-entity", type=str, help="wanDB entity name")
g.add_argument("--wandb-project", type=str, help="wanDB project name")
# fmt: on


def main(args: argparse.Namespace):
    logger = get_logger("train")

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed)

    logger.info(f'[+] Load Train Dataset from "{args.train_dataset_pattern}"')
    train_dataset = SampleDataset(args.input_dimension, args.num_classes)
    logger.info(f'[+] Load Valid Dataset from "{args.valid_dataset_pattern}"')
    valid_dataset = SampleDataset(args.input_dimension, args.num_classes)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    if args.gpus is None:
        args.gpus = torch.cuda.device_count()
    num_parallels = max(args.gpus, 1)
    logger.info(f"[+] GPUs: {num_parallels}")
    total_steps = len(train_dataloader) * args.epochs // args.accumulate_grad_batches // num_parallels

    classification = SimpleClassification(
        input_dimension=args.input_dimension,
        num_classes=args.num_classes,
        total_steps=total_steps,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_rate=args.warmup_rate,
    )

    if args.pretrained_model_path:
        logger.info(f'[+] Load Model from "{args.pretrained_model_path}"')
        classification.model.load_state_dict(torch.load(args.pretrained_model_path))

    logger.info(f"[+] Start Training")
    train_loggers = [TensorBoardLogger(args.output_dir, "logs")]
    if args.wandb_project:
        train_loggers.append(
            WandbLogger(
                name=args.wandb_run_name or os.path.basename(args.output_dir),
                project=args.wandb_project,
                entity=args.wandb_entity,
                save_dir=args.output_dir,
            )
        )
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[
            ModelCheckpoint(monitor="val/loss", save_last=True, save_weights_only=True),
            LearningRateMonitor(logging_interval="step"),
        ],
        strategy="auto",
        accelerator="auto",
        devices=num_parallels,
    )
    trainer.fit(classification, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
