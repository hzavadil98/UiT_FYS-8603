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
from src import Four_view_single_featurizer, View_Cancer_Dataloader


def main():
    """
    Training single view featurizer models - 4 of them each on a specific "modality#
    """
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
        torch.set_float32_matmul_precision("high")

    train_transform = T.Compose(
        [
            T.ToImage(),
            # T.RandomRotation(degrees=10),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(
                mean=[781.0543, 781.0543, 781.0543],
                std=[1537.8235, 1537.8235, 1537.8235],
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
                mean=[781.0543, 781.0543, 781.0543],
                std=[1537.8235, 1537.8235, 1537.8235],
            ),
            # T.RandomAdjustSharpness(sharpness_factor=1, p=1),
            # T.Normalize(
            #    mean=[0.5, 0.5, 0.5],
            #    std=[0.7, 0.7, 0.7],
            # ),
        ]
    )

    views = ["CC", "MLO", "CC", "MLO"]
    lateralities = ["L", "L", "R", "R"]

    for i in range(4):
        dataloader = View_Cancer_Dataloader(
            root_folder=root_folder,
            annotation_csv="modified_breast-level_annotations.csv",
            imagefolder_path="New_512",
            batch_size=32,
            num_workers=8,
            view=views[i],
            laterality=lateralities[i],
            train_transform=train_transform,
            transform=transform,
        )
        # dataloader.train_dataset.plot(0)

        model = Four_view_single_featurizer(
            num_class=5, drop=0.5, learning_rate=1e-4, view=i
        )
        # check_dataloader_passes_model(dataloader, model)

        wandb_logger = WandbLogger(
            project="Single_View_Featurizers", log_model="best", name=f"View_{i}"
        )
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
            # limit_train_batches=3,  # Only 5 training batches per epoch
            # limit_val_batches=2,
            # log_every_n_steps=1,
        )

        # Train
        trainer.fit(model, dataloader)
        # Load best weights
        print(
            f"Finished training, loading the best epoch: {checkpoint_callback.best_model_path}"
        )
        model = Four_view_single_featurizer.load_from_checkpoint(
            checkpoint_callback.best_model_path
        )
        # Test
        trainer.test(model, dataloader)

        # Finish wandb run
        wandb.finish()

    return 0


if __name__ == "__main__":
    main()
