import os  # Add os import

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
from src import Breast_Cancer_Dataloader, Synthetic_2v_Dataloader, TwoViewCNN

"""
train_transform = T.Compose(
    [
        T.Normalize(mean=[781.0543], std=[1537.8235]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
    ]
)
transform = T.Compose(
    [
        T.Normalize(mean=[781.0543], std=[1537.8235]),
    ]
)"""


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

    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.GaussianNoise(0.1, 0.1),
        ]
    )
    dataloader = Synthetic_2v_Dataloader(
        n_samples=[3000, 1000, 1000],
        train_transform=train_transform,
        transform=None,
        batch_size=32,
    )
    ##########################################################################################################

    # dataloader = Breast_Cancer_Dataloader(
    #    root_folder="/storage/Mammo/",
    #    annotation_csv="modified_breast-level_annotations.csv",
    #    imagefolder_path="New_512",
    #    batch_size=32,
    #    num_workers=4,
    #    train_transform=train_transform,
    #    transform=transform,
    # )
    ##########################################################################################################
    model = TwoViewCNN(
        num_classes=4,
        task=2,
        num_views=2,
        input_channels=1,
        resnext_inplanes=16,
        learning_rate=1e-3,
        scheduler_patience=5,  # Or any other value you prefer
        scheduler_factor=0.2,  # Or any other value you prefer
    )
    run_name = f"Synth_data_task_{model.task}"

    # Set WANDB_CODE_DIR to save all code in the current directory and subdirectories
    os.environ["WANDB_CODE_DIR"] = "."
    wandb_logger = WandbLogger(project="Synthetic data", log_model=True, name=run_name)
    # wandb_logger.watch(model, log="best", log_freq=5)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=run_name + "_best",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    early_stopping = EarlyStopping(monitor="val_loss", patience=15, mode="min")

    # Add profiler
    # profiler = PyTorchProfiler(dirpath="./profiler_logs", filename="profile") # Temporarily disable profiler
    trainer = pl.Trainer(
        max_epochs=50,
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
