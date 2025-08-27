import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as T
import wandb
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from src import Breast_Cancer_Dataloader, Two_view_model


def main():
    """
    Training a two views model - CC,MLO but lateralities mixed together
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
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ]
    )

    # transform = T.Compose(
    #    [
    # T.ToImage(),
    # T.ToDtype(torch.float32, scale=True),
    # T.Normalize(
    #    mean=[781.0543, 781.0543, 781.0543],
    #    std=[1537.8235, 1537.8235, 1537.8235],
    # ),
    # T.RandomAdjustSharpness(sharpness_factor=1, p=1),
    # T.Normalize(
    #    mean=[0.5, 0.5, 0.5],
    #    std=[0.7, 0.7, 0.7],
    # ),
    #    ]
    # )

    imagefolder_path = "New_512"
    image_format = "dicom"
    norm_kind = "zscore"
    batch_size = 16
    task = 1  # 1 for cancer, 2 for density

    dataloader = Breast_Cancer_Dataloader(
        root_folder=root_folder,
        annotation_csv="modified_breast-level_annotations.csv",
        imagefolder_path=imagefolder_path,
        image_format=image_format,
        norm_kind=norm_kind,
        batch_size=batch_size,
        num_workers=8,
        train_transform=train_transform,
        transform=None,
        task=task,  # 2 for density classification
    )
    # dataloader.train_dataset.plot(0)

    model = Two_view_model(
        num_class=5,
        weights_file="checkpoints/One_view_resnet.ckpt",
        drop=0.5,
        learning_rate=1e-5,
        task=task,  # 2 for density classification
    )
    # model = Mirai_two_view_model(
    #    num_class=5,
    #    drop=0.3,
    #    learning_rate=1e-4,
    #    task=1,  # 1 for cancer classification
    # )
    # check_dataloader_passes_model(dataloader, model)

    wandb_logger = WandbLogger(
        project="Two_view_one_branch_model",
        log_model="True",
        name=f"Resnet_{imagefolder_path}_{norm_kind}_{'cancer' if task == 1 else 'density'}",
    )
    # wandb_logger.watch(model, log="all", log_freq=10)
    wandb_logger.experiment.config.update(
        {
            "imagefolder_path": imagefolder_path,
            "image_format": image_format,
            "norm_kind": norm_kind,
            "batch_size": batch_size,
        }
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="2v1b_{imagefolder_path}_{norm_kind}_task:{task:02d}-epoch:{epoch:02d}",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor="val_loss", patience=8, mode="min")

    # figure out if running with mps or gpu or cpu

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        accumulate_grad_batches=8,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        #    limit_train_batches=3,  # Only 3 training batches per epoch
        #    limit_val_batches=2,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, dataloader)
    # Load best weights
    print(
        f"Finished training, loading the best epoch: {checkpoint_callback.best_model_path}"
    )

    model = Two_view_model.load_from_checkpoint(checkpoint_callback.best_model_path)
    # Test
    trainer.test(model, dataloader)

    # Finish wandb run
    wandb.finish()

    return 0


if __name__ == "__main__":
    main()
