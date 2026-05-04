import os
from pathlib import Path

print("[DEBUG] Importing pytorch_lightning...", flush=True)
import pytorch_lightning as pl

print("[DEBUG] ✓ pytorch_lightning imported", flush=True)

print("[DEBUG] Importing torch...", flush=True)
import torch

print("[DEBUG] ✓ torch imported", flush=True)

print("[DEBUG] Importing torchvision...", flush=True)
import torchvision.transforms.v2 as T

print("[DEBUG] ✓ torchvision imported", flush=True)

print("[DEBUG] Importing pytorch_lightning callbacks...", flush=True)
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

print("[DEBUG] ✓ callbacks imported", flush=True)

print("[DEBUG] Importing WandbLogger...", flush=True)
from pytorch_lightning.loggers import WandbLogger

print("[DEBUG] ✓ WandbLogger imported", flush=True)

print("[DEBUG] Importing wandb...", flush=True)
import wandb

print("[DEBUG] ✓ wandb imported", flush=True)

print("[DEBUG] Importing AJIVE from python_packages...", flush=True)
from python_packages import AJIVE

print("[DEBUG] ✓ AJIVE imported", flush=True)

print("[DEBUG] Importing src models...", flush=True)
from src import Single_view_AJIVE_heads, Single_view_model, View_Cancer_Dataloader

print("[DEBUG] ✓ src models imported", flush=True)


def get_runtime_config():
    """Select the best available device and matching data root."""
    if torch.backends.mps.is_available():
        return {
            "root_folder": "/Users/jazav7774/Data/Mammo/",
            "working_folder": "/",
            "accelerator": "mps",
            "devices": 1,
        }

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        return {
            "root_folder": "/storage/Mammo/",
            "working_folder": "/root/UiT_FYS-8603/",
            "accelerator": "gpu",
            "devices": torch.cuda.device_count(),
        }

    return {
        "root_folder": "/Users/jazav7774/Data/Mammo/",
        "working_folder": "/",
        "accelerator": "cpu",
        "devices": 1,
    }


def build_train_transform():
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
        ]
    )


def load_saved_ajive_inputs(feature_dir: Path):
    """Load the saved train/val/test feature blocks used to fit AJIVE."""
    if not feature_dir.exists():
        raise FileNotFoundError(
            f"Could not find saved feature directory: {feature_dir}"
        )

    splits = {}
    for split_name in ("train", "val", "test"):
        split_path = feature_dir / f"all_data_{split_name}.pt"
        splits[split_name] = torch.load(split_path, map_location="cpu")

    return splits


def fit_ajive_model(train_cancer_features, train_density_features):
    """Fit the shared AJIVE model used by both head-training runs."""
    ajive_model = AJIVE(init_signal_ranks=[15, 15], n_jobs=1, center=True)
    ajive_model.fit([train_cancer_features, train_density_features])
    return ajive_model


def build_head_model(base_model, ajive_model, active_head, num_class):
    """Create a fresh AJIVE-head module for a specific active head."""
    return Single_view_AJIVE_heads(
        model=base_model,
        ajive_model=ajive_model,
        num_class=num_class,
        active_head=active_head,
        hidden_dim=128,
        drop=0.3,
        learning_rate=1e-3,
    )


def train_and_test_heads(
    task_name,
    base_model,
    ajive_model,
    dataloader,
    accelerator,
    devices,
    head_names,
    max_epochs,
    project_name,
    root_folder,
    imagefolder_path,
    image_format,
    norm_kind,
    batch_size,
    num_workers,
):
    """Train the three AJIVE heads sequentially and log each as a separate wandb run."""
    os.environ["WANDB_CODE_DIR"] = "."

    model = build_head_model(
        base_model=base_model,
        ajive_model=ajive_model,
        active_head=head_names[0],
        num_class=base_model.hparams.num_class,
    )

    print(f"\nTraining AJIVE heads for {task_name}...")
    print("=" * 70)

    for head_name in head_names:
        print(f"\nTraining {task_name} AJIVE head: {head_name}")
        print("-" * 70)

        # Create a separate wandb run for each head
        wandb_logger = WandbLogger(
            project=project_name,
            log_model=True,
            name=f"AJIVE_heads_{task_name}_{head_name}",
            settings=wandb.Settings(init_timeout=300),
        )

        wandb_logger.experiment.config.update(
            {
                "task_name": task_name,
                "head_name": head_name,
                "root_folder": root_folder,
                "imagefolder_path": imagefolder_path,
                "image_format": image_format,
                "norm_kind": norm_kind,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "max_epochs": max_epochs,
                "ajive_init_signal_ranks": [15, 15],
            }
        )

        model.set_active_head(head_name)

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/ajive_heads_{task_name}/",
            filename=f"ajive_heads_{task_name}_{head_name}-epoch:{{epoch:02d}}",
            save_top_k=1,
            monitor=f"val_{head_name}_loss",
            mode="min",
            save_last=True,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stopping = EarlyStopping(
            monitor=f"val_{head_name}_loss",
            patience=4,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
            log_every_n_steps=10,
            accumulate_grad_batches=4,
            enable_progress_bar=True,
        )

        trainer.fit(model, dataloader)

        print(f"Finished {task_name} {head_name} head training.")
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")

        if checkpoint_callback.best_model_path:
            model = Single_view_AJIVE_heads.load_from_checkpoint(
                checkpoint_callback.best_model_path,
                model=base_model,
                ajive_model=ajive_model,
                num_class=base_model.hparams.num_class,
                learning_rate=1e-3,
                hidden_dim=128,
                drop=0.3,
                active_head=head_name,
            )

        trainer.test(model, dataloader)

        # Finish the current run before starting the next head
        wandb.finish()

    return model


def main():
    """Train the cancer and density AJIVE head models and log them to wandb."""
    print("Setting up runtime configuration...")
    runtime = get_runtime_config()
    root_folder = runtime["root_folder"]
    # working_folder = runtime["working_folder"]
    # os.chdir(working_folder)
    accelerator = runtime["accelerator"]
    devices = runtime["devices"]

    if accelerator in {"mps", "gpu"}:
        torch.set_float32_matmul_precision("high")

    train_transform = build_train_transform()

    imagefolder_path = "images_png_396"
    image_format = "png"
    norm_kind = "dataset_zscore"
    batch_size = 32
    num_workers = 4
    task = 1

    dataloader = View_Cancer_Dataloader(
        root_folder=root_folder,
        annotation_csv="modified_breast-level_annotations.csv",
        imagefolder_path=imagefolder_path,
        image_format=image_format,
        norm_kind=norm_kind,
        batch_size=batch_size,
        num_workers=num_workers,
        train_transform=train_transform,
        task=task,
        use_train_sampler=True,
    )
    print("flag 1")
    print(os.getcwd())
    print(os.listdir(os.getcwd()))
    model_cancer = Single_view_model.load_from_checkpoint(
        "artifacts/model-ln6ychcp:v0/model.ckpt"
    )
    model_density = Single_view_model.load_from_checkpoint(
        "artifacts/model-vjzmam1e:v0/model.ckpt"
    )
    print("flag 2")

    feature_dir = Path("saved_features/One_View_Canc_vs_Dens")
    feature_splits = load_saved_ajive_inputs(feature_dir)

    train_data = feature_splits["train"]
    train_cancer_features = train_data["cancer_features"].cpu().numpy()
    train_density_features = train_data["dens_features"].cpu().numpy()

    print("Fitting shared AJIVE model from saved training features...")
    ajive_model = fit_ajive_model(train_cancer_features, train_density_features)
    print("AJIVE fitting completed.")

    head_names = ["joint", "individual", "both"]
    max_epochs = 12
    project_name = "AJIVE_Mammo_heads"

    cancer_ajive_model = train_and_test_heads(
        task_name="cancer",
        base_model=model_cancer,
        ajive_model=ajive_model,
        dataloader=dataloader,
        accelerator=accelerator,
        devices=devices,
        head_names=head_names,
        max_epochs=max_epochs,
        project_name=project_name,
        root_folder=root_folder,
        imagefolder_path=imagefolder_path,
        image_format=image_format,
        norm_kind=norm_kind,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    density_ajive_model = train_and_test_heads(
        task_name="density",
        base_model=model_density,
        ajive_model=ajive_model,
        dataloader=dataloader,
        accelerator=accelerator,
        devices=devices,
        head_names=head_names,
        max_epochs=max_epochs,
        project_name=project_name,
        root_folder=root_folder,
        imagefolder_path=imagefolder_path,
        image_format=image_format,
        norm_kind=norm_kind,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("\nTraining complete.")
    print(
        "Cancer AJIVE model and density AJIVE model were both trained and logged to wandb."
    )

    return cancer_ajive_model, density_ajive_model


if __name__ == "__main__":
    main()
