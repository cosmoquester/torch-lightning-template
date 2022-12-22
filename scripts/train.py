import argparse
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
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
g.add_argument("--batch-size", type=int, default=128, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=256, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help="the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=10, help="the number of training epochs")
g.add_argument("--max-learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--min-learning-rate", type=float, default=1e-5, help="min Learning rate")
g.add_argument("--warmup-rate", type=float, default=0.05, help="warmup step rate")
g.add_argument("--gpus", type=int, default=0, help="the number of gpus")
g.add_argument("--logging-interval", type=int, default=10, help="logging interval")
g.add_argument("--evaluate-interval", type=int, default=50, help="validation interval")
g.add_argument("--seed", type=int, default=42, help="random seed")

g = parser.add_argument_group("Wandb Options")
g.add_argument("--wandb-run-name", type=str, help="wanDB run name")
g.add_argument("--wandb-entity", type=str, help="wanDB entity name")
g.add_argument("--wandb-project", type=str, help="wanDB project name")
# fmt: on


def main(args: argparse.Namespace):
    logger = get_logger("train")

    os.makedirs(args.output_dir)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed)

    logger.info(f"[+] GPU: {args.gpus}")

    logger.info(f'[+] Load Train Dataset from "{args.train_dataset_pattern}"')
    train_dataset = SampleDataset(args.input_dimension, args.num_classes)
    logger.info(f'[+] Load Valid Dataset from "{args.valid_dataset_pattern}"')
    valid_dataset = SampleDataset(args.input_dimension, args.num_classes)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    total_steps = len(train_dataloader) * args.epochs

    model_dir = os.path.join(args.output_dir, "models")
    os.makedirs(model_dir)
    classification = SimpleClassification(
        input_dimension=args.input_dimension,
        num_classes=args.num_classes,
        total_steps=total_steps,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        warmup_rate=args.warmup_rate,
        model_save_dir=model_dir,
    )

    if args.pretrained_model_path:
        logger.info(f'[+] Load Model from "{args.pretrained_model_path}"')
        classification.model.load_state_dict(torch.load(args.pretrained_model_path))

    logger.info(f"[+] Start Training")
    train_loggers = [TensorBoardLogger(args.output_dir, "", "logs")]
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
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.evaluate_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        gpus=args.gpus,
    )
    trainer.fit(classification, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
