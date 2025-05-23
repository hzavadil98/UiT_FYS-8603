import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as T
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import Synthetic_2v_Dataloader, TwoViewCNN


def main():
    """
    Training single view featurizer models - 4 of them each on a specific "modality#
    """
    # Recognizes if running on my mac or on the server - sets the root_folder and accelerator
    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        torch.set_float32_matmul_precision("high")

    transform = T.Compose(
        [
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.GaussianNoise(0.1, 0.1),
        ]
    )

    for i in range(4):
        dataloader = Synthetic_2v_Dataloader(
            transform=transform,
        )

        model = TwoViewCNN(num_classes=3)

        wandb_logger = WandbLogger(project="Synthetic data", log_model="best")
        wandb_logger.watch(model, log="all", log_freq=5)

        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",
            filename=f"best_epoch_view:{i}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stopping = EarlyStopping(monitor="val_loss", patience=12, mode="min")

        # figure out if running with mps or gpu or cpu

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator=accelerator,
            devices=devices,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
            log_every_n_steps=5,
        )

        # Train
        trainer.fit(model, dataloader)
        # Load best weights
        print(
            f"Finished training, loading the best epoch: {checkpoint_callback.best_model_path}"
        )
        model = TwoViewCNN.load_from_checkpoint(checkpoint_callback.best_model_path)
        # Test
        trainer.test(model, dataloader)

        # Finish wandb run
        wandb.finish()

    return 0
