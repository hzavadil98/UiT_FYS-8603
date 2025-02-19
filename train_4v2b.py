import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

import wandb
from src import Four_view_two_branch_model, Patient_Cancer_Dataloader


def check_dataloader_passes_model(dataloader, model):
    batch = next(iter(dataloader.train_dataloader()))
    x, y = batch

    y_left, y_right = model.forward(x)

    print(f"data passed ok, outputs: {y_left.shape}, {y_right.shape}")

    return 0


def main():
    # Recognizes if running on my mac or on the server - sets the root_folder and accelerator
    if torch.backends.mps.is_available():
        root_folder = (
            "/Users/jazav7774/Library/CloudStorage/OneDrive-UiTOffice365/Data/Mammo/"
        )
        accelerator = "mps"
        devices = 1
    elif torch.cuda.is_available():
        root_folder = "/storage/VinDR-data/"
        accelerator = "gpu"
        devices = torch.cuda.device_count()

    dataloader = Patient_Cancer_Dataloader(
        root_folder=root_folder,
        annotation_csv="modified_breast-level_annotations.csv",
        imagefolder_path="New_512",
        batch_size=16,
        num_workers=8,
    )
    # dataloader.train_dataset.plot(0)

    model = Four_view_two_branch_model(num_class=5)
    # check_dataloader_passes_model(dataloader, model)

    wandb_logger = WandbLogger(project="Four_view_two_branch_model", log_model="all")
    wandb_logger.watch(model, log="all", log_freq=100)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best_epoch-{epoch:02d}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor="val_loss", patience=8, mode="min")

    # figure out if running with mps or gpu or cpu

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
    )

    # Train
    trainer.fit(model, dataloader)
    # Load best weights
    print(
        f"Finished training, loading the best epoch: {checkpoint_callback.best_model_path}"
    )
    model = Four_view_two_branch_model.load_from_checkpoint(
        checkpoint_callback.best_model_path
    )
    # Test
    trainer.test(model, dataloader)

    # Finish wandb run
    wandb.finish()

    return 0


if __name__ == "__main__":
    main()
