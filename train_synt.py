import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as T
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler  # Add this import

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
        # torch.cuda.empty_cache()

    print(f"Using {accelerator} with {devices} devices for training.")

    transform = T.Compose(
        [
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.GaussianNoise(0.1, 0.1),
        ]
    )

    # Initialize DataLoader once before the loop if data is the same for all runs
    dataloader = Synthetic_2v_Dataloader(
        n_samples=[5000, 1000, 1000], transform=transform, batch_size=8
    )

    model = TwoViewCNN(num_classes=3)

    wandb_logger = WandbLogger(project="Synthetic data", log_model="best")
    wandb_logger.watch(model, log="all", log_freq=1)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best_epoch",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor="val_loss", patience=12, mode="min")

    # Add profiler
    profiler = PyTorchProfiler(dirpath="./profiler_logs", filename="profile")
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=1,
        profiler=profiler,  # Add profiler to trainer
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


if __name__ == "__main__":
    main()
