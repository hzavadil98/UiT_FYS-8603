import argparse
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

NUM_CANCER_CLASSES = {"birads": 5, "diagnosis": 3, "binary": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a single-view VinDR model")
    parser.add_argument(
        "--task", type=int, default=2, choices=[1, 2], help="1 = cancer, 2 = density"
    )
    parser.add_argument(
        "--cancer-label-type",
        default="birads",
        choices=list(NUM_CANCER_CLASSES),
        help="Cancer label mapping (only relevant for task 1)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 epoch on a few batches, no wandb - to check the pipeline runs",
    )
    return parser.parse_args()


def main():
    """
    Training CC specific and MLO specific models
    """
    args = parse_args()
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
    task = args.task
    cancer_label_type = args.cancer_label_type
    # suffix keeps runs/checkpoints of non-default label types apart
    label_suffix = (
        f"_{cancer_label_type}" if task == 1 and cancer_label_type != "birads" else ""
    )

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
        cancer_label_type=cancer_label_type,
    )
    # dataloader.train_dataset.plot(0)

    model = Single_view_model(
        num_class=NUM_CANCER_CLASSES[cancer_label_type] if task == 1 else 4,
        drop=0.4,
        weights_file=1,
        learning_rate=1e-4,
        task=task,
    )
    # check_dataloader_passes_model(dataloader, model)
    os.environ["WANDB_CODE_DIR"] = "."
    if args.smoke:
        wandb_logger = False
    else:
        wandb_logger = WandbLogger(
            project="Single_View_Models",
            log_model=True,
            name=f"Model_single_view_{'cancer' if task == 1 else 'density'}{label_suffix}",
        )

        wandb_logger.experiment.config.update(
            {
                "imagefolder_path": imagefolder_path,
                "image_format": image_format,
                "norm_kind": norm_kind,
                "batch_size": batch_size,
                "cancer_label_type": cancer_label_type,
            }
        )

    checkpoint_filename = f"model_single_view_{imagefolder_path}_{norm_kind}_task{task:02d}{label_suffix}-epoch:{{epoch:02d}}"

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=checkpoint_filename,
        save_top_k=1,
        monitor="val_f1",
        mode="max",
        save_last=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stopping = EarlyStopping(monitor="val_f1", patience=8, mode="max")

    trainer = pl.Trainer(
        max_epochs=1 if args.smoke else 100,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping]
        + ([] if args.smoke else [lr_monitor]),  # lr monitor needs a logger
        log_every_n_steps=10,
        accumulate_grad_batches=4,
        limit_train_batches=3 if args.smoke else None,
        limit_val_batches=2 if args.smoke else None,
        limit_test_batches=2 if args.smoke else None,
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
    if not args.smoke:
        wandb.finish()

    return 0


if __name__ == "__main__":
    main()
