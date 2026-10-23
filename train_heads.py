import argparse
import os
from pathlib import Path

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
from python_packages import AJIVE
from src import Single_view_AJIVE_heads, Single_view_model, View_Cancer_Dataloader


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


def fit_ajive_model(train_cancer_features, train_density_features, ranks=[15, 15]):
    """Fit the shared AJIVE model used by both head-training runs."""
    ajive_model = AJIVE(init_signal_ranks=ranks, n_jobs=1, center=True)
    ajive_model.fit([train_cancer_features, train_density_features])
    return ajive_model


def build_head_model(base_model, ajive_model, active_head, num_class):
    """Create a fresh AJIVE-head module for a specific active head."""
    return Single_view_AJIVE_heads(
        model=base_model,
        ajive_model=ajive_model,
        num_class=num_class,
        active_head=active_head,
        hidden_dim=56,
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
    ajive_ranks,
    cancer_label_type,
    smoke=False,
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
        if smoke:
            wandb_logger = False
        else:
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
                    "ajive_init_signal_ranks": list(ajive_ranks),
                    "cancer_label_type": cancer_label_type,
                }
            )

        model.set_active_head(head_name)

        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/ajive_heads_{task_name}/",
            filename=f"ajive_heads_{task_name}_{head_name}-epoch:{{epoch:02d}}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            save_last=False,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=4,
            mode="min",
        )

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, early_stopping]
            + ([] if smoke else [lr_monitor]),  # lr monitor needs a logger
            log_every_n_steps=10,
            accumulate_grad_batches=4,
            enable_progress_bar=True,
            limit_train_batches=3 if smoke else None,
            limit_val_batches=2 if smoke else None,
            limit_test_batches=2 if smoke else None,
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
                hidden_dim=56,
                drop=0.3,
                active_head=head_name,
            )

        trainer.test(model, dataloader)

        # Finish the current run before starting the next head
        if not smoke:
            wandb.finish()

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train AJIVE heads on VinDR")
    parser.add_argument(
        "--cancer-ckpt", default="artifacts/model-5opqyl7e:v0/model.ckpt"
    )
    parser.add_argument(
        "--density-ckpt", default="artifacts/model-kb9kz79s:v0/model.ckpt"
    )
    parser.add_argument("--feature-dir", default="saved_features/One_View_Canc_vs_Dens")
    parser.add_argument(
        "--cancer-label-type",
        default="birads",
        choices=["birads", "diagnosis", "binary"],
    )
    parser.add_argument("--ranks", type=int, nargs=2, default=[15, 15])
    parser.add_argument("--project", default="AJIVE_Mammo_heads")
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="1 epoch on a few batches, no wandb - to check the pipeline runs",
    )
    return parser.parse_args()


def main():
    """Train the cancer and density AJIVE head models and log them to wandb."""
    args = parse_args()
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
    cancer_label_type = args.cancer_label_type
    # suffix keeps runs/checkpoints of non-default label types apart
    label_suffix = "" if cancer_label_type == "birads" else f"_{cancer_label_type}"

    def build_dataloader(task):
        # the weighted train sampler balances the classes of the given task
        return View_Cancer_Dataloader(
            root_folder=root_folder,
            annotation_csv="modified_breast-level_annotations.csv",
            imagefolder_path=imagefolder_path,
            image_format=image_format,
            norm_kind=norm_kind,
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=train_transform,
            task=task,
            cancer_label_type=cancer_label_type,
            use_train_sampler=True,
        )

    cancer_dataloader = build_dataloader(task=1)
    density_dataloader = build_dataloader(task=2)

    model_cancer = Single_view_model.load_from_checkpoint(args.cancer_ckpt)
    model_density = Single_view_model.load_from_checkpoint(args.density_ckpt)

    feature_dir = Path(args.feature_dir)
    feature_splits = load_saved_ajive_inputs(feature_dir)

    train_data = feature_splits["train"]
    train_cancer_features = train_data["cancer_features"].cpu().numpy()
    train_density_features = train_data["dens_features"].cpu().numpy()

    print("Fitting shared AJIVE model from saved training features...")
    ajive_model = fit_ajive_model(
        train_cancer_features, train_density_features, ranks=args.ranks
    )
    print("AJIVE fitting completed.")

    head_names = ["joint", "individual", "both"]
    max_epochs = 1 if args.smoke else args.max_epochs
    project_name = args.project

    common_kwargs = dict(
        ajive_model=ajive_model,
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
        ajive_ranks=args.ranks,
        cancer_label_type=cancer_label_type,
        smoke=args.smoke,
    )

    cancer_ajive_model = train_and_test_heads(
        task_name=f"cancer{label_suffix}",
        base_model=model_cancer,
        dataloader=cancer_dataloader,
        **common_kwargs,
    )

    density_ajive_model = train_and_test_heads(
        task_name=f"density{label_suffix}",
        base_model=model_density,
        dataloader=density_dataloader,
        **common_kwargs,
    )

    print("\nTraining complete.")
    print(
        "Cancer AJIVE model and density AJIVE model were both trained and logged to wandb."
    )

    return cancer_ajive_model, density_ajive_model


if __name__ == "__main__":
    main()
