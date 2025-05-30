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

print("Using custom TwoViewCNN model for synthetic data.")


def main():
    """
    Training single view featurizer models - 4 of them each on a specific "modality"
    """
    mps_available = torch.backends.mps.is_available()

    if mps_available:
        accelerator = "mps"
        devices = 1
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        torch.set_float32_matmul_precision("high")
        # torch.cuda.empty_cache()
    else:
        accelerator = "cpu"  # Default to CPU if neither is available
        devices = 1

    transform = T.Compose(
        [
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.GaussianNoise(0.1, 0.1),
        ]
    )

    # Initialize DataLoader once before the loop if data is the same for all runs
    dataloader = Synthetic_2v_Dataloader(
        n_samples=[5000, 1000, 1000], transform=transform, batch_size=16
    )
    ##########################################################################################################
    """
    train_transform = T.Compose(
        [
            T.ToImage(),
            # T.RandomRotation(degrees=10),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=[781.0543],
                std=[1537.8235],
            ),
            # T.RandomAdjustSharpness(sharpness_factor=1, p=1),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            # T.RandomRotation(degrees=10),
            # T.Normalize(
            #    mean=[0.5, 0.5, 0.5],
            #    std=[0.7, 0.7, 0.7],
            # ),
        ]
    )

    transform = T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=[781.0543],
                std=[1537.8235],
            ),
            # T.RandomAdjustSharpness(sharpness_factor=1, p=1),
            # T.Normalize(
            #    mean=[0.5, 0.5, 0.5],
            #    std=[0.7, 0.7, 0.7],
            # ),
        ]
    )

    dataloader = Breast_Cancer_Dataloader(
        root_folder="/storage/Mammo/",
        annotation_csv="modified_breast-level_annotations.csv",
        imagefolder_path="New_512",
        batch_size=16,
        num_workers=8,
        train_transform=train_transform,
        transform=transform,
    )
    """
    ##########################################################################################################
    model = TwoViewCNN(
        num_classes=3, task=1, num_views=2, input_channels=1, resnext_inplanes=16
    )

    wandb_logger = WandbLogger(
        project="Synthetic data", log_model="best", name="Synthetic data v2 task 1"
    )
    # wandb_logger.watch(model, log="all", log_freq=1) # Temporarily disable watch

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="best_epoch",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stopping = EarlyStopping(monitor="val_loss", patience=4, mode="min")

    # Add profiler
    # profiler = PyTorchProfiler(dirpath="./profiler_logs", filename="profile") # Temporarily disable profiler
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=10,
        # profiler=profiler,  # Temporarily disable profiler
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
