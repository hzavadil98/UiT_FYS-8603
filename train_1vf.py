import os

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
from src import Single_view_model, View_Cancer_Dataloader


def main():
    """
    Training CC specific and MLO specific models
    """
    # Recognizes if running on my mac or on the server - sets the root_folder and accelerator
    if torch.backends.mps.is_available():
        root_folder = "/Users/jazav7774/Data/Mammo/"
        accelerator = "mps"
        devices = 1
    elif torch.cuda.is_available():
        root_folder = "/storage/Mammo/"
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        torch.set_float32_matmul_precision("high")

    train_transform = T.Compose(
        [
            # T.ToImage(),
            # T.RandomRotation(degrees=10),
            # T.ToDtype(torch.float32, scale=True),
            # T.Normalize(
            #    mean=[781.0543, 781.0543, 781.0543],
            #    std=[1537.8235, 1537.8235, 1537.8235],
            # ),
            # T.RandomAdjustSharpness(sharpness_factor=1, p=1),
            # T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            #            T.Resize(396),              #resizes the shorter side to 396 pixels while keeping aspect ratio (660x396), total numel similar to 512*512
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ]
    )

    imagefolder_path = "images_png_396"
    image_format = "png"
    norm_kind = "dataset_zscore"
    batch_size = 32
    task = 1

    dataloader = View_Cancer_Dataloader(
        root_folder=root_folder,
        annotation_csv="modified_breast-level_annotations.csv",
        imagefolder_path=imagefolder_path,
        image_format=image_format,
        norm_kind=norm_kind,
        batch_size=batch_size,
        num_workers=4,
        train_transform=train_transform,
        task=task,
    )
    # dataloader.train_dataset.plot(0)

    model = Single_view_model(num_class=5, drop=0.4, learning_rate=1e-4, task=task)
    # check_dataloader_passes_model(dataloader, model)
    os.environ["WANDB_CODE_DIR"] = "."
    wandb_logger = WandbLogger(
        project="Single_View_Models",
        log_model=True,
        name=f"Model_CC+MLO_{'cancer' if task == 1 else 'density'}",
    )

    wandb_logger.experiment.config.update(
        {
            "imagefolder_path": imagefolder_path,
            "image_format": image_format,
            "norm_kind": norm_kind,
            "batch_size": batch_size,
        }
    )

    checkpoint_filename = f"model_CC+MLO_{imagefolder_path}_{norm_kind}_task{task:02d}-epoch:{{epoch:02d}}"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor="val_loss", patience=8, mode="min")

    trainer = pl.Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=10,
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
    model = Single_view_model.load_from_checkpoint(checkpoint_callback.best_model_path)
    # Test
    trainer.test(model, dataloader)

    # Finish wandb run
    wandb.finish()

    return 0


if __name__ == "__main__":
    main()
