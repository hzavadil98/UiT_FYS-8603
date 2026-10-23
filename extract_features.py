import argparse
import os

import torch
import tqdm

from src import Single_view_model, View_Cancer_Dataloader
from train_heads import get_runtime_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract ResNet features of the cancer and density models for AJIVE"
    )
    parser.add_argument("--cancer-ckpt", required=True)
    parser.add_argument(
        "--density-ckpt", default="artifacts/model-kb9kz79s:v0/model.ckpt"
    )
    parser.add_argument(
        "--cancer-label-type",
        default="birads",
        choices=["birads", "diagnosis", "binary"],
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only featurize a few batches per split (saved to <save-dir>_smoke)",
    )
    return parser.parse_args()


def get_featurized_data(dataloader, model, model1, device, max_batches=None):
    """
    Get the full featurized dataset from a dataloader.
    """
    model.to(device)
    model.eval()
    model1.to(device)
    model1.eval()
    with torch.no_grad():
        all_in_features = []
        all_in_features1 = []
        logits = []
        logits1 = []

        all_labels = []
        all_labels1 = []

        for batch_idx, batch in enumerate(
            tqdm.tqdm(dataloader, desc="Featurizing dataset")
        ):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x, y1, y2 = batch
            x = x.to(device)
            all_labels.append(y1)
            all_labels1.append(y2)

            x0 = model.get_resnet_outputs(x)
            x1 = model1.get_resnet_outputs(x)

            all_in_features.append(x0)
            all_in_features1.append(x1)

            logits.append(model.fc(model.resnet.fc(x0)))
            logits1.append(model1.fc(model1.resnet.fc(x1)))

        all_in_features = torch.cat(all_in_features, dim=0)
        all_in_features1 = torch.cat(all_in_features1, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_labels1 = torch.cat(all_labels1, dim=0)
        logits = torch.cat(logits, dim=0)
        logits1 = torch.cat(logits1, dim=0)

    return all_in_features, all_labels, logits, all_in_features1, all_labels1, logits1


def process_and_save_all_splits(
    breast_loader, model, model_dens, device, save_dir, max_batches=None
):
    """
    Process train, validation, and test dataloaders and save features and labels.
    """
    os.makedirs(save_dir, exist_ok=True)

    splits = {
        "train": breast_loader.train_dataloader(),
        "val": breast_loader.val_dataloader(),
        "test": breast_loader.test_dataloader(),
    }

    for split_name, dataloader in splits.items():
        print(f"Processing {split_name} split...")

        (
            cancer_features,
            cancer_labels,
            cancer_logits,
            dens_features,
            dens_labels,
            dens_logits,
        ) = get_featurized_data(
            dataloader, model, model_dens, device, max_batches=max_batches
        )

        # Same keys as the existing saved_features/One_View_Canc_vs_Dens caches
        torch.save(
            {
                "cancer_features": cancer_features.cpu(),
                "cancer_labels": cancer_labels.cpu(),
                "cancer_logits": cancer_logits.cpu(),
                "dens_features": dens_features.cpu(),
                "dens_labels": dens_labels.cpu(),
                "dens_logits": dens_logits.cpu(),
            },
            f"{save_dir}/all_data_{split_name}.pt",
        )

        print(f"Saved {split_name} split with shapes:")
        print(f"  Cancer features: {cancer_features.shape}")
        print(f"  Cancer labels: {cancer_labels.shape}")
        print(f"  Density features: {dens_features.shape}")
        print(f"  Density labels: {dens_labels.shape}")
        print()


def main():
    args = parse_args()
    runtime = get_runtime_config()
    device = {"gpu": "cuda"}.get(runtime["accelerator"], runtime["accelerator"])

    # No sampler and no augmentation - every training image featurized exactly once
    dataloader = View_Cancer_Dataloader(
        root_folder=runtime["root_folder"],
        annotation_csv="modified_breast-level_annotations.csv",
        imagefolder_path="images_png_396",
        image_format="png",
        norm_kind="dataset_zscore",
        batch_size=32,
        num_workers=4,
        task=1,
        cancer_label_type=args.cancer_label_type,
        use_train_sampler=False,
    )

    model_cancer = Single_view_model.load_from_checkpoint(
        args.cancer_ckpt, map_location="cpu"
    )
    model_density = Single_view_model.load_from_checkpoint(
        args.density_ckpt, map_location="cpu"
    )

    save_dir = f"{args.save_dir.rstrip('/')}_smoke" if args.smoke else args.save_dir
    process_and_save_all_splits(
        dataloader,
        model_cancer,
        model_density,
        device,
        save_dir,
        max_batches=2 if args.smoke else None,
    )


if __name__ == "__main__":
    main()
