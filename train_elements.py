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
from src import Single_Element_Dataloader, Single_view_model


def main():
    """Train a single-view model on the synthetic single-element dataset.

    task=1 -> shape classification (4 classes)
    task=2 -> texture classification (3 classes)
    """
    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count()
        torch.set_float32_matmul_precision("high")
    else:
        accelerator = "cpu"
        devices = 1

    train_transform = T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ]
    )

    batch_size = 32
    task = 2  # 1: shape (4 classes), 2: texture (3 classes)

    dataloader = Single_Element_Dataloader(
        batch_size=batch_size,
        num_workers=4,
        n_train=5000,
        n_val=1000,
        n_test=1000,
        img_size=224,
        element_n=1,
        element_size=96,
        element_size_delta=24,
        allowed_shapes=["square", "circle", "triangle", "plus"],
        allowed_colors=["red", "green", "blue"],
        allowed_textures=["solid", "spots_polka", "stripes_diagonal"],
        element_seed=42,
        loc_seed=123,
        train_transform=train_transform,
        transform=None,
        use_train_sampler=True,
        sampler_target="shape" if task == 1 else "texture",
    )

    model = Single_view_model(
        num_class=4 if task == 1 else 3,
        drop=0.4,
        weights_file=1,
        learning_rate=1e-4,
        task=task,
    )

    os.environ["WANDB_CODE_DIR"] = "."
    wandb_logger = WandbLogger(
        project="Single_Element_Models",
        log_model=True,
        name=f"Model_single_element_{'shape' if task == 1 else 'texture'}",
    )

    wandb_logger.experiment.config.update(
        {
            "batch_size": batch_size,
            "task": task,
            "n_train": 5000,
            "n_val": 1000,
            "n_test": 1000,
            "img_size": 224,
            "element_n": 1,
            "element_size": 96,
            "element_size_delta": 24,
            "allowed_shapes": ["square", "circle", "triangle", "plus"],
            "allowed_colors": ["red", "green", "blue"],
            "allowed_textures": ["solid", "spots_polka", "stripes_diagonal"],
        }
    )

    checkpoint_filename = f"model_single_element_task{task:02d}-epoch:{{epoch:02d}}"

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
        max_epochs=100,
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor, early_stopping],
        log_every_n_steps=10,
        accumulate_grad_batches=4,
    )

    trainer.fit(model, dataloader)

    print(
        f"Finished training, loading the best epoch: {checkpoint_callback.best_model_path}"
    )
    model = Single_view_model.load_from_checkpoint(checkpoint_callback.best_model_path)

    trainer.test(model, dataloader)

    wandb.finish()

    return 0


if __name__ == "__main__":
    main()
