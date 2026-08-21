# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "bokeh<=3.9.0",
#     "imageio==2.37.3",
#     "marimo>=0.23.8",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "pydicom==3.0.2",
#     "pytorch-lightning==2.6.5",
#     "scikit-learn==1.8.0",
#     "seaborn==0.13.2",
#     "statsmodels==0.14.6",
#     "torch==2.12.0",
#     "torchvision==0.27.0",
#     "tqdm==4.67.3",
# ]
# ///

import marimo

__generated_with = "0.23.8"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Setup
    """)
    return


@app.cell
def _():
    import ast
    import os
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import tqdm
    from matplotlib.patches import Rectangle
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    from src import Single_view_AJIVE_heads, Single_view_model, View_Cancer_Dataloader

    sys.path.insert(0, os.path.abspath("python_packages"))
    from python_packages import AJIVE  # Ensure AJIVE classes are importable.

    return (
        Rectangle,
        Single_view_AJIVE_heads,
        Single_view_model,
        View_Cancer_Dataloader,
        accuracy_score,
        ast,
        balanced_accuracy_score,
        f1_score,
        np,
        os,
        pd,
        plt,
        sns,
        torch,
        tqdm,
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo, os, torch):
    device_default = "mps" if torch.backends.mps.is_available() else "cpu"
    cache_dir = os.path.join("saved_features", "vinDr_annot")

    root_folder = mo.ui.text(
        value="/Users/jazav7774/Data/Mammo/",
        label="Data root folder",
    )
    annotation_csv = mo.ui.text(
        value="modified_breast-level_annotations.csv",
        label="Breast-level annotation CSV",
    )
    detection_csv = mo.ui.text(
        value="/Users/jazav7774/Data/Mammo/vindr_detection_v1_folds.csv",
        label="Detection annotations CSV",
    )
    imagefolder_path = mo.ui.text(
        value="images_png_396",
        label="Image folder (under data root)",
    )

    model_cancer_ckpt = mo.ui.text(
        value="artifacts/model-5opqyl7e:v0/model.ckpt",
        label="Cancer model checkpoint",
    )
    model_density_ckpt = mo.ui.text(
        value="artifacts/model-kb9kz79s:v0/model.ckpt",
        label="Density model checkpoint",
    )
    ajive_cancer_ckpt = mo.ui.text(
        value="artifacts/model-x541jdxj:v0/model.ckpt",
        label="Cancer AJIVE head checkpoint",
    )
    ajive_density_ckpt = mo.ui.text(
        value="artifacts/model-ngta50dn:v0/model.ckpt",
        label="Density AJIVE head checkpoint",
    )
    device_select = mo.ui.dropdown(
        options=["mps", "cpu", "gpu"],
        value=device_default,
        label="Device",
    )

    setup_ui = mo.vstack(
        [
            root_folder,
            annotation_csv,
            detection_csv,
            imagefolder_path,
            model_cancer_ckpt,
            model_density_ckpt,
            ajive_cancer_ckpt,
            ajive_density_ckpt,
            device_select,
        ]
    )
    setup_ui
    return (
        ajive_cancer_ckpt,
        ajive_density_ckpt,
        annotation_csv,
        cache_dir,
        detection_csv,
        device_select,
        imagefolder_path,
        model_cancer_ckpt,
        model_density_ckpt,
        root_folder,
    )


@app.cell
def _(
    Single_view_AJIVE_heads,
    Single_view_model,
    View_Cancer_Dataloader,
    ajive_cancer_ckpt,
    ajive_density_ckpt,
    annotation_csv,
    ast,
    detection_csv,
    device_select,
    imagefolder_path,
    model_cancer_ckpt,
    model_density_ckpt,
    pd,
    root_folder,
    torch,
):
    dataloader = View_Cancer_Dataloader(
        root_folder=root_folder.value,
        annotation_csv=annotation_csv.value,
        imagefolder_path=imagefolder_path.value,
        image_format="png",
        norm_kind="dataset_zscore",
        batch_size=32,
        num_workers=1,
        task=1,
        use_train_sampler=False,
    )

    model_cancer = Single_view_model.load_from_checkpoint(
        model_cancer_ckpt.value,
        map_location="cpu",
    )
    model_density = Single_view_model.load_from_checkpoint(
        model_density_ckpt.value,
        map_location="cpu",
    )

    def load_ajive_head_checkpoint(path: str, backbone, map_location: str = "cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        hp = ckpt["hyper_parameters"]

        view_index = backbone.task - 1
        model = Single_view_AJIVE_heads.load_from_checkpoint(
            path,
            map_location=map_location,
            weights_only=False,
            model=backbone,
            ajive_model=hp["ajive_model"],
            num_class=hp["num_class"],
            learning_rate=hp.get("learning_rate", 1e-3),
            hidden_dim=hp.get("hidden_dim", 56),
            drop=hp.get("drop", 0.3),
            active_head=hp.get("active_head", "both"),
            view_index=view_index,
        )
        return model

    cancer_ajive_model = load_ajive_head_checkpoint(
        ajive_cancer_ckpt.value,
        model_cancer,
    )
    density_ajive_model = load_ajive_head_checkpoint(
        ajive_density_ckpt.value,
        model_density,
    )

    device = device_select.value
    model_cancer = model_cancer.to(device).eval()
    model_density = model_density.to(device).eval()
    cancer_ajive_model = cancer_ajive_model.to(device).eval()
    density_ajive_model = density_ajive_model.to(device).eval()

    annotations = pd.read_csv(detection_csv.value, low_memory=False)
    annot_test = annotations[
        (annotations["image_id"].isin(dataloader.test_dataset.image_ids))
        & (annotations["finding_categories"] != "['No Finding']")
    ].copy()
    annot_test["finding_categories_list"] = annot_test["finding_categories"].apply(
        ast.literal_eval
    )
    annot_test = annot_test.explode("finding_categories_list").reset_index(drop=True)
    annot_test = annot_test.rename(
        columns={"finding_categories_list": "finding_category"}
    )

    annot_test
    return (
        annot_test,
        cancer_ajive_model,
        dataloader,
        density_ajive_model,
        device,
        model_cancer,
        model_density,
    )


@app.cell
def _(annot_test, pd, plt):
    if 1:
        # Create a stacked bar chart showing BI-RADS distribution
        _fig, _ax = plt.subplots(figsize=(14, 8))

        # Get category counts for sorting
        category_counts = (
            annot_test["finding_category"].value_counts().sort_values(ascending=True)
        )

        # Stacked bar chart with BI-RADS distribution
        birads_counts = pd.crosstab(
            annot_test["finding_category"], annot_test["breast_birads"]
        )

        # Sort by total count
        birads_counts = birads_counts.loc[category_counts.index]

        birads_counts.plot(
            kind="barh",
            stacked=True,
            ax=_ax,
            colormap="viridis",
            edgecolor="white",
            linewidth=0.5,
        )
        _ax.set_xlabel("Number of Occurrences", fontsize=12)
        _ax.set_ylabel("Finding Category", fontsize=12)
        _ax.set_title(
            "BI-RADS Distribution by Finding Category", fontsize=14, fontweight="bold"
        )
        _ax.legend(title="Breast BI-RADS", bbox_to_anchor=(1.05, 1), loc="upper left")
        _ax.grid(axis="x", alpha=0.3)

        # Create summary table
        birads_percentages = birads_counts.div(birads_counts.sum(axis=1), axis=0) * 100

        # Prepare table data - one total column and percentages for each BI-RADS
        table_data = []
        table_headers = ["Category", "#n"]
        for birads in birads_counts.columns:
            table_headers.append(f"{birads}")

        for category in reversed(birads_counts.index):
            row = [category]  # Keep full category name
            total = birads_counts.loc[category].sum()
            row.append(f"{total}")
            for birads in birads_counts.columns:
                pct = birads_percentages.loc[category, birads]
                row.append(f"{pct:.1f}%")
            table_data.append(row)

        # Add a total row
        total_row = ["Total"]
        grand_total = birads_counts.sum().sum()
        total_row.append(f"{grand_total}")
        for birads in birads_counts.columns:
            total_pct = (birads_counts[birads].sum() / grand_total) * 100
            total_row.append(f"{total_pct:.1f}%")
        table_data.append(total_row)

        # Add table to the plot
        table = _ax.table(
            cellText=table_data,
            colLabels=table_headers,
            cellLoc="center",
            loc="lower right",
            bbox=[0.34, 0.02, 1.5 * 0.43, 1.5 * 0.35],  # [x, y, width, height]
            cellColours=[["lightgray"] * len(table_headers)] * len(table_data),
            colColours=["steelblue"] * len(table_headers),
        )

        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)

        # Style the header and adjust column widths
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor("steelblue")
            elif i == len(table_data):  # Total row
                cell.set_text_props(weight="bold")
                cell.set_facecolor("lightsteelblue")
            else:
                cell.set_facecolor("whitesmoke" if i % 2 == 0 else "white")

            # Make first column (Category) wider
            if j == 0:
                cell.set_width(0.39)
            elif j == 1:
                cell.set_width(0.13)
            else:
                cell.set_width(0.2)

    _fig
    return


@app.cell(hide_code=True)
def _(Rectangle, np, pd, plt, torch, tqdm):
    def visualize_grad_cam(
        model,
        image_tensor,
        image_label=None,
        target_class=None,
        suptitle=None,
        cmap="jet",
        denormalize=True,
        mean=108.3328,
        std=69.6431,
        show_suptitle=True,
    ):
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.shape[0] > 1:
            image_tensor = image_tensor[0:1]

        grad_cam, predicted_class = model.generate_grad_cam(image_tensor, target_class)
        image_np = image_tensor[0].detach().permute(1, 2, 0).cpu().numpy()

        if denormalize:
            image_np = image_np * std + mean
            image_np = image_np / 255.0
            image_np = np.clip(image_np, 0, 1)

        grad_cam_resized = torch.nn.functional.interpolate(
            grad_cam.unsqueeze(1),
            size=(image_np.shape[0], image_np.shape[1]),
            mode="bilinear",
            align_corners=False,
        )
        grad_cam_resized = grad_cam_resized.squeeze(1)[0].numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_np, cmap="gray")
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        im1 = axes[1].imshow(grad_cam_resized, cmap=cmap)
        axes[1].set_title("Grad-CAM Heatmap")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1])

        axes[2].imshow(image_np, cmap="gray")
        im2 = axes[2].imshow(grad_cam_resized, cmap=cmap, alpha=0.5)
        axes[2].set_title(
            f"Overlay (Pred: Class {predicted_class.item()}, True: Class {image_label})"
        )
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2])

        if suptitle is not None:
            plt.suptitle(suptitle)
        plt.tight_layout()

        return fig

    def calculate_bbox_probability_mass(saliency_map, annot_row):
        height, width = saliency_map.shape

        saliency_prob = (
            saliency_map / saliency_map.sum()
            if saliency_map.sum() > 0
            else saliency_map
        )

        bbox_mask = np.zeros((height, width), dtype=bool)

        x_min = int(annot_row["resized_xmin"] * width / 912)
        y_min = int(annot_row["resized_ymin"] * height / 1520)
        x_max = int(annot_row["resized_xmax"] * width / 912)
        y_max = int(annot_row["resized_ymax"] * height / 1520)

        x_min = max(0, min(x_min, width - 1))
        x_max = max(0, min(x_max, width))
        y_min = max(0, min(y_min, height - 1))
        y_max = max(0, min(y_max, height))

        bbox_mask[y_min:y_max, x_min:x_max] = True

        if saliency_map.sum() == 0:
            return bbox_mask, 0.0, 0.0, 0.0, 0.0, False

        bbox_mass = saliency_prob[bbox_mask].sum()
        bbox_area = bbox_mask.sum()
        total_area = saliency_map.size
        expected_mass = bbox_area / total_area if total_area > 0 else 0
        enrichment = bbox_mass / expected_mass if expected_mass > 0 else 0

        mean_inside = saliency_prob[bbox_mask].mean() if bbox_mask.sum() > 0 else 0
        mean_outside = saliency_prob[~bbox_mask].mean() if (~bbox_mask).sum() > 0 else 0
        mass_ratio = mean_inside / mean_outside if mean_outside > 0 else np.inf

        max_loc = np.unravel_index(saliency_map.argmax(), saliency_map.shape)
        pointing_game = bbox_mask[max_loc]

        return (
            bbox_mask,
            enrichment,
            mean_inside,
            mean_outside,
            mass_ratio,
            pointing_game,
        )

    def plot_grad_cam_with_bbox(
        model,
        annot_row,
        dataset,
        head_name=None,
        target_class=None,
        use_true_class=False,
        cmap="jet",
        denormalize=True,
        mean=108.3328,
        std=69.6431,
        show_suptitle=True,
    ):
        dataloader_idx = np.where(dataset.image_ids == annot_row["image_id"])[0][0]
        image_tensor, label1, label2 = dataset[dataloader_idx]

        y1_label = int(label1.item() if torch.is_tensor(label1) else label1)
        y2_label = int(label2.item() if torch.is_tensor(label2) else label2)
        true_label = y1_label if getattr(model, "task", 1) == 1 else y2_label
        if use_true_class and target_class is None:
            target_class = torch.tensor(
                [true_label], device=next(model.parameters()).device
            )

        device = next(model.parameters()).device
        if head_name is None:
            cam, predicted_class = model.generate_grad_cam(
                image_tensor.unsqueeze(0).to(device),
                target_class=target_class,
            )
        else:
            cam, predicted_class = model.generate_grad_cam(
                image_tensor.unsqueeze(0).to(device),
                target_class=target_class,
                head_name=head_name,
            )

        cam = cam.detach().cpu()
        image_np = image_tensor.detach().permute(1, 2, 0).cpu().numpy()
        if denormalize:
            image_np = image_np * std + mean
            image_np = image_np / 255.0
            image_np = np.clip(image_np, 0, 1)

        cam_resized = (
            torch.nn.functional.interpolate(
                cam.unsqueeze(1),
                size=(image_np.shape[0], image_np.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)[0]
            .numpy()
        )

        bbox_mask, enrichment, mean_inside, mean_outside, mass_ratio, pointing_game = (
            calculate_bbox_probability_mass(cam_resized, annot_row)
        )

        bbox_coords = np.argwhere(bbox_mask)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image_np, cmap="gray")
        axes[0].set_title(
            f"Original Image\nTrue: BI-RADS {y1_label}, Density {y2_label}"
        )
        axes[0].axis("off")

        if bbox_coords.size > 0:
            y_min, x_min = bbox_coords.min(axis=0)
            y_max, x_max = bbox_coords.max(axis=0)
            rect_kwargs = dict(
                linewidth=2, edgecolor="cyan", facecolor="none", linestyle="--"
            )
            axes[0].add_patch(
                Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, **rect_kwargs)
            )

        im1 = axes[1].imshow(cam_resized, cmap=cmap)
        axes[1].set_title(f"Grad-CAM Heatmap\nEnrichment: {enrichment:.2f}x")
        axes[1].axis("off")
        if bbox_coords.size > 0:
            axes[1].add_patch(
                Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, **rect_kwargs)
            )
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        axes[2].imshow(image_np, cmap="gray")
        im2 = axes[2].imshow(cam_resized, cmap=cmap, alpha=0.5)
        axes[2].set_title(
            f"Overlay\nPred: Class {predicted_class.item()}, Pointing Game: {pointing_game}"
        )
        axes[2].axis("off")
        if bbox_coords.size > 0:
            axes[2].add_patch(
                Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, **rect_kwargs)
            )
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        if show_suptitle:
            fig.suptitle(
                f"Grad-CAM with Bounding Box\nAnnotation: {annot_row['finding_category']}",
                fontsize=14,
                fontweight="bold",
            )
        plt.tight_layout()

        bbox_metrics = {
            "enrichment": enrichment,
            "mean_inside": mean_inside,
            "mean_outside": mean_outside,
            "mass_ratio": mass_ratio,
            "pointing_game": bool(pointing_game),
        }
        return fig, predicted_class.item(), bbox_metrics

    def _get_gradcam_saliency(
        model,
        dataset,
        annot_row,
        head_name=None,
        use_true_class=False,
    ):
        dataloader_idx = np.where(dataset.image_ids == annot_row["image_id"])[0][0]
        image_tensor, label1, label2 = dataset[dataloader_idx]
        y1_label = int(label1.item() if torch.is_tensor(label1) else label1)
        y2_label = int(label2.item() if torch.is_tensor(label2) else label2)

        model = model.eval()
        task_label = y1_label if getattr(model, "task", 1) == 1 else y2_label
        target_class = task_label if use_true_class else None
        if target_class is not None:
            target_class = torch.tensor(
                [target_class], device=next(model.parameters()).device
            )

        if head_name is None or not hasattr(model, "head_names"):
            cam, predicted_class = model.generate_grad_cam(
                image_tensor.unsqueeze(0).to(next(model.parameters()).device),
                target_class=target_class,
            )
        else:
            cam, predicted_class = model.generate_grad_cam(
                image_tensor.unsqueeze(0).to(next(model.parameters()).device),
                target_class=target_class,
                head_name=head_name,
            )

        saliency_map = cam.squeeze(0).detach().cpu().numpy()
        return saliency_map, predicted_class.item(), y1_label, y2_label

    def process_all_annotations_gradcam_ajive_head(
        model,
        dataset,
        annot_df,
        head_name=None,
        model_name=None,
        use_true_class=False,
        pointing_game_aggregate="image",
    ):
        results = []
        problematic_indices = []
        model_name = model_name or model.__class__.__name__
        head_label = head_name if head_name is not None else "original"
        desc = f"{model_name}-{head_label} (true_class={use_true_class})"

        for idx, row in tqdm.tqdm(annot_df.iterrows(), total=len(annot_df), desc=desc):
            saliency_map, pred_class, y1_label, y2_label = _get_gradcam_saliency(
                model,
                dataset,
                row,
                head_name=head_name,
                use_true_class=use_true_class,
            )

            if saliency_map.sum() == 0:
                problematic_indices.append(idx)

            (
                bbox_mask,
                enrichment,
                mean_inside,
                mean_outside,
                mass_ratio,
                pointing_game,
            ) = calculate_bbox_probability_mass(saliency_map, row)

            correct = (
                pred_class == y1_label
                if getattr(model, "task", 1) == 1
                else (pred_class == y2_label)
            )

            results.append(
                {
                    "model_name": model_name,
                    "head_name": head_label,
                    "patient_id": row["patient_id"],
                    "series_id": row["series_id"],
                    "image_id": row["image_id"],
                    "view": row["view"],
                    "laterality": row["laterality"],
                    "finding_category": row["finding_category"],
                    "BI-RADS": y1_label,
                    "Density": y2_label,
                    "predicted_class": pred_class,
                    "correct": correct,
                    "enrichment": enrichment,
                    "mean_inside": mean_inside,
                    "mean_outside": mean_outside,
                    "mass_ratio": mass_ratio,
                    "pointing_game": bool(pointing_game),
                    "bbox_mask": bbox_mask,
                }
            )

        results_df = pd.DataFrame(results)
        if pointing_game_aggregate not in {"image", "row"}:
            raise ValueError("pointing_game_aggregate must be 'image' or 'row'")
        if (
            pointing_game_aggregate == "image"
            and not results_df.empty
            and "image_id" in results_df.columns
            and "pointing_game" in results_df.columns
        ):
            results_df["pointing_game"] = (
                results_df.groupby("image_id")["pointing_game"]
                .transform("max")
                .astype(bool)
            )
        if problematic_indices:
            print(f"{desc}: {len(problematic_indices)} zero-mass saliency maps.")
        return results_df

    def plot_localization_metrics(
        results_list,
        method_labels=None,
        stratify=None,
        correct_only=False,
        top_k=None,
        birads_values=None,
        metrics=("enrichment", "mass_ratio", "pointing_game"),
        figsize=(20, 6),
        group_width=0.8,
        y_limit_quantile=0.95,
        y_limit_scale=1.15,
    ):
        if results_list is None or len(results_list) == 0:
            raise ValueError("results_list must contain at least one dataframe")
        if method_labels is None:
            method_labels = [f"Method {i + 1}" for i in range(len(results_list))]
        if len(method_labels) != len(results_list):
            raise ValueError("method_labels length must match results_list length")

        if stratify is None:
            stratify_cols = []
        elif isinstance(stratify, (list, tuple)):
            stratify_cols = list(stratify)
        else:
            stratify_cols = [stratify]

        if len(stratify_cols) > 2:
            raise ValueError("At most two stratify columns are supported")

        if correct_only:

            def _correct_mask(df):
                if "correct" in df.columns:
                    return df["correct"].astype(bool)
                return df["BI-RADS"] == df["predicted_class"]

            dfs = [df.loc[_correct_mask(df)].copy() for df in results_list]
        else:
            dfs = [df.copy() for df in results_list]

        ref_df = dfs[0]
        if "finding_category" in stratify_cols and top_k is not None:
            top_categories = (
                ref_df["finding_category"].value_counts().head(top_k).index.tolist()
            )
            dfs = [df[df["finding_category"].isin(top_categories)].copy() for df in dfs]
            ref_df = dfs[0]

        def _label_value(col, value):
            if col == "BI-RADS":
                try:
                    return f"BI-{int(value) + 1}"
                except Exception:
                    return f"BI-{value}"
            return str(value)

        def _major_order(column, frame):
            if column == "BI-RADS" and birads_values is not None:
                return list(birads_values)
            return frame[column].value_counts().index.tolist()

        if not stratify_cols:
            group_keys = [tuple()]
            group_labels = ["All"]
        elif len(stratify_cols) == 1:
            major_col = stratify_cols[0]
            major_order = _major_order(major_col, ref_df)
            group_keys = [(major,) for major in major_order]
            group_labels = [
                f"{_label_value(major_col, major)} (n={len(ref_df[ref_df[major_col] == major])})"
                for major in major_order
            ]
        else:
            major_col, minor_col = stratify_cols
            major_order = _major_order(major_col, ref_df)
            if minor_col == "BI-RADS" and birads_values is not None:
                minor_order = list(birads_values)
            elif minor_col == "BI-RADS":
                minor_order = sorted(ref_df[minor_col].unique().tolist())
            else:
                minor_order = ref_df[minor_col].value_counts().index.tolist()
            group_keys = [
                (major, minor) for major in major_order for minor in minor_order
            ]
            group_labels = [
                f"{_label_value(major_col, major)}\n{_label_value(minor_col, minor)}"
                f" (n={len(ref_df[(ref_df[major_col] == major) & (ref_df[minor_col] == minor)])})"
                for major in major_order
                for minor in minor_order
            ]

        def _subset(df, key):
            if not stratify_cols:
                return df
            if len(stratify_cols) == 1:
                return df[df[stratify_cols[0]] == key[0]]
            return df[
                (df[stratify_cols[0]] == key[0]) & (df[stratify_cols[1]] == key[1])
            ]

        n_groups = len(group_keys)
        n_methods = len(dfs)
        x = np.arange(n_groups)
        box_width = group_width / max(n_methods, 1)
        offset_start = -group_width / 2 + box_width / 2
        colors = plt.cm.Set2(np.linspace(0, 1, n_methods))

        fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
        if len(metrics) == 1:
            axes = [axes]

        for plot_idx, metric in enumerate(metrics):
            ax = axes[plot_idx]
            if metric in ("enrichment", "mass_ratio"):
                for j, df in enumerate(dfs):
                    data = [_subset(df, key)[metric].values for key in group_keys]
                    pos = x + offset_start + j * box_width
                    bp = ax.boxplot(
                        data,
                        positions=pos,
                        widths=box_width,
                        patch_artist=True,
                        showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="red", markersize=4),
                        boxprops=dict(linewidth=0.8),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        medianprops=dict(linewidth=0.8),
                    )
                    for patch in bp["boxes"]:
                        patch.set_facecolor(colors[j])
                        patch.set_alpha(0.75)

                ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
                ax.grid(axis="y", alpha=0.3)
                ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5)

                all_vals = []
                for df in dfs:
                    for key in group_keys:
                        vals = _subset(df, key)[metric].values
                        if len(vals) > 0:
                            all_vals.append(vals)
                if all_vals:
                    all_vals = np.concatenate(all_vals)
                    all_vals = all_vals[np.isfinite(all_vals)]
                    if all_vals.size > 0:
                        y_max = np.quantile(all_vals, y_limit_quantile) * y_limit_scale
                        if y_max > 0:
                            ax.set_ylim(0, y_max)
            elif metric == "pointing_game":
                max_total = 0
                for j, df in enumerate(dfs):
                    successes = []
                    failures = []
                    totals = []
                    for key in group_keys:
                        sub = _subset(df, key)
                        success = int(sub["pointing_game"].sum())
                        total = len(sub)
                        successes.append(success)
                        failures.append(total - success)
                        totals.append(total)
                        max_total = max(max_total, total)

                    pos = x + offset_start + j * box_width
                    for idx in range(len(group_keys)):
                        ax.bar(
                            pos[idx],
                            successes[idx],
                            width=box_width,
                            color=colors[j],
                            alpha=0.8,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        ax.bar(
                            pos[idx],
                            failures[idx],
                            width=box_width,
                            bottom=successes[idx],
                            color=colors[j],
                            alpha=0.3,
                            edgecolor="black",
                            linewidth=0.5,
                        )
                        if totals[idx] > 0:
                            pct = successes[idx] / totals[idx]
                            ax.text(
                                pos[idx],
                                successes[idx] + failures[idx] + max_total * 0.02,
                                f"{pct:.0%}",
                                ha="center",
                                va="bottom",
                                fontsize=6,
                                rotation=45,
                            )

                ax.set_ylabel("Number of Samples", fontsize=10)
                ax.set_xticks(x)
                ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
                ax.grid(axis="y", alpha=0.3)
                if max_total > 0:
                    ax.set_ylim(0, max_total * y_limit_scale)
            else:
                raise ValueError(f"Unsupported metric: {metric}")

            ax.set_title(
                metric.replace("_", " ").title(), fontsize=11, fontweight="bold"
            )

        handles = [
            plt.Line2D([0], [0], color=colors[j], lw=6) for j in range(n_methods)
        ]
        fig.legend(handles, method_labels, loc="upper right", fontsize=9)
        fig.tight_layout()
        return fig, axes

    def compute_gradcam_data(
        model,
        annot_row,
        dataset,
        head_name=None,
        use_true_class=False,
        mean=108.3328,
        std=69.6431,
    ):
        dataloader_idx = np.where(dataset.image_ids == annot_row["image_id"])[0][0]
        image_tensor, label1, label2 = dataset[dataloader_idx]
        y1_label = int(label1.item() if torch.is_tensor(label1) else label1)
        y2_label = int(label2.item() if torch.is_tensor(label2) else label2)
        task_label = y1_label if getattr(model, "task", 1) == 1 else y2_label
        device = next(model.parameters()).device
        target_class = (
            torch.tensor([task_label], device=device) if use_true_class else None
        )

        if head_name is None or not hasattr(model, "head_names"):
            cam, predicted_class = model.generate_grad_cam(
                image_tensor.unsqueeze(0).to(device), target_class=target_class
            )
        else:
            cam, predicted_class = model.generate_grad_cam(
                image_tensor.unsqueeze(0).to(device),
                target_class=target_class,
                head_name=head_name,
            )

        cam = cam.detach().cpu()
        image_np = image_tensor.detach().permute(1, 2, 0).cpu().numpy()
        image_np = np.clip(image_np * std + mean, 0, 255) / 255.0

        H, W = image_np.shape[:2]
        cam_resized = (
            torch.nn.functional.interpolate(
                cam.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            )
            .squeeze(1)[0]
            .numpy()
        )

        _, enrichment, *_ = calculate_bbox_probability_mass(cam_resized, annot_row)
        return (
            image_np,
            cam_resized,
            predicted_class.item(),
            enrichment,
            y1_label,
            y2_label,
        )

    return (
        compute_gradcam_data,
        plot_localization_metrics,
        process_all_annotations_gradcam_ajive_head,
    )


@app.function
def head_names(model):
    if hasattr(model, "head_names"):
        return list(model.head_names)
    if hasattr(model, "heads"):
        return list(model.heads.keys())
    return []


@app.cell
def _(mo):
    mo.md("""
    # Visualizations
    """)
    return


@app.cell
def _(
    annot_test,
    cache_dir,
    cancer_ajive_model,
    dataloader,
    density_ajive_model,
    mo,
    model_cancer,
    model_density,
    os,
    pd,
    process_all_annotations_gradcam_ajive_head,
):
    results_pred_path = os.path.join(cache_dir, "gradcam_results_pred.pkl")
    results_true_path = os.path.join(cache_dir, "gradcam_results_true.pkl")

    if os.path.exists(results_pred_path) and os.path.exists(results_true_path):
        results_list_pred = pd.read_pickle(results_pred_path)
        results_list_true = pd.read_pickle(results_true_path)
    else:
        gradcam_models = [
            ("original_cancer", model_cancer, None),
            ("original_density", model_density, None),
        ]

        for _head_name in head_names(cancer_ajive_model):
            gradcam_models.append(
                (f"cancer_ajive_{_head_name}", cancer_ajive_model, _head_name),
            )
        for _head_name in head_names(density_ajive_model):
            gradcam_models.append(
                (f"density_ajive_{_head_name}", density_ajive_model, _head_name),
            )

        results_list_pred = []
        results_list_true = []
        for _model_name, _model, _head_name in gradcam_models:
            df_pred = process_all_annotations_gradcam_ajive_head(
                _model,
                dataloader.test_dataset,
                annot_test.copy(),
                head_name=_head_name,
                model_name=_model_name,
                use_true_class=False,
            )
            df_true = process_all_annotations_gradcam_ajive_head(
                _model,
                dataloader.test_dataset,
                annot_test.copy(),
                head_name=_head_name,
                model_name=_model_name,
                use_true_class=True,
            )
            results_list_pred.append(df_pred)
            results_list_true.append(df_true)

        pd.to_pickle(results_list_pred, results_pred_path)
        pd.to_pickle(results_list_true, results_true_path)

    results_by_model_pred = {
        df["model_name"].iloc[0]: df for df in results_list_pred if not df.empty
    }
    results_by_model_true = {
        df["model_name"].iloc[0]: df for df in results_list_true if not df.empty
    }

    mo.md(
        f"Computed {len(results_list_pred)} result tables for predicted and true class Grad-CAM."
    )
    return (
        results_by_model_pred,
        results_by_model_true,
        results_list_pred,
        results_list_true,
    )


@app.cell(hide_code=True)
def _(
    cache_dir,
    cancer_ajive_model,
    dataloader,
    density_ajive_model,
    device,
    mo,
    model_cancer,
    model_density,
    np,
    os,
    pd,
    torch,
    tqdm,
):
    _all_preds_path = os.path.join(cache_dir, "all_test_preds.pkl")

    if os.path.exists(_all_preds_path):
        all_model_preds = pd.read_pickle(_all_preds_path)
    else:
        _test_dl = dataloader.test_dataloader()

        _head_names_list = (
            list(cancer_ajive_model.head_names)
            if hasattr(cancer_ajive_model, "head_names")
            else list(cancer_ajive_model.heads.keys())
        )

        _pred_collectors = {"original_cancer": [], "original_density": []}
        for _hn in _head_names_list:
            _pred_collectors[f"cancer_ajive_{_hn}"] = []
            _pred_collectors[f"density_ajive_{_hn}"] = []

        _labels_cancer, _labels_density = [], []

        with torch.no_grad():
            for _batch in tqdm.tqdm(_test_dl, desc="Full-test predictions"):
                _images, _lbl_cancer, _lbl_density = _batch
                _images = _images.to(device)

                _pred_collectors["original_cancer"].extend(
                    model_cancer(_images).argmax(dim=1).cpu().numpy()
                )
                _pred_collectors["original_density"].extend(
                    model_density(_images).argmax(dim=1).cpu().numpy()
                )

                for _hn in _head_names_list:
                    cancer_ajive_model.set_active_head(_hn)
                    _pred_collectors[f"cancer_ajive_{_hn}"].extend(
                        cancer_ajive_model(_images, head_name=_hn)
                        .argmax(dim=1)
                        .cpu()
                        .numpy()
                    )
                for _hn in _head_names_list:
                    density_ajive_model.set_active_head(_hn)
                    _pred_collectors[f"density_ajive_{_hn}"].extend(
                        density_ajive_model(_images, head_name=_hn)
                        .argmax(dim=1)
                        .cpu()
                        .numpy()
                    )

                _labels_cancer.extend(_lbl_cancer.cpu().numpy())
                _labels_density.extend(_lbl_density.cpu().numpy())

        _labels_cancer = np.array(_labels_cancer)
        _labels_density = np.array(_labels_density)

        all_model_preds = {}
        for _mname, _preds in _pred_collectors.items():
            _preds = np.array(_preds)
            _true = _labels_cancer if "cancer" in _mname else _labels_density
            all_model_preds[_mname] = pd.DataFrame(
                {
                    "predicted_class": _preds,
                    "BI-RADS": _labels_cancer,
                    "Density": _labels_density,
                    "correct": _preds == _true,
                }
            )

        pd.to_pickle(all_model_preds, _all_preds_path)

    mo.md(
        f"Full-test predictions loaded for **{len(all_model_preds)}** models ({next(iter(all_model_preds.values())).__len__()} samples each)."
    )
    return (all_model_preds,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Model Prediction Agreement
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    agreement_task_select = mo.ui.dropdown(
        options=["Cancer", "Density"], value="Cancer", label="Task"
    )
    agreement_task_select
    return (agreement_task_select,)


@app.cell(hide_code=True)
def _(agreement_task_select, all_model_preds, np, pd, plt, sns):
    _task = agreement_task_select.value

    _cancer_names = sorted(
        [k for k in all_model_preds if "cancer" in k],
        key=lambda x: (0 if x == "original_cancer" else 1),
    )
    _density_names = sorted(
        [k for k in all_model_preds if "density" in k],
        key=lambda x: (0 if x == "original_density" else 1),
    )
    _names = _cancer_names if _task == "Cancer" else _density_names
    _class_col = "BI-RADS" if _task == "Cancer" else "Density"

    _short_label = {
        "original_cancer": "Original",
        "cancer_ajive_individual": "Indiv.",
        "cancer_ajive_joint": "Joint",
        "cancer_ajive_both": "Both",
        "original_density": "Original",
        "density_ajive_individual": "Indiv.",
        "density_ajive_joint": "Joint",
        "density_ajive_both": "Both",
    }

    def _pairwise_agreement(names):
        _preds = {n: all_model_preds[n]["predicted_class"].to_numpy() for n in names}
        _mat = np.array(
            [[(_preds[a] == _preds[b]).mean() for b in names] for a in names]
        )
        _lbl = [_short_label.get(n, n) for n in names]
        return pd.DataFrame(_mat, index=_lbl, columns=_lbl)

    def _correctness_matrix(names):
        _mat = np.column_stack(
            [all_model_preds[n]["correct"].to_numpy().astype(int) for n in names]
        )
        _order = np.argsort(_mat.sum(axis=1))
        return _mat[_order], [_short_label.get(n, n) for n in names]

    def _class_accuracy(names, class_col):
        _ref = all_model_preds[names[0]]
        _classes = sorted(_ref[class_col].unique())
        _fmt = (
            (lambda c: f"BI-{int(c) + 1}")
            if class_col == "BI-RADS"
            else (lambda c: str(c))
        )
        _rows = {}
        for n in names:
            _df = all_model_preds[n]
            _rows[_short_label.get(n, n)] = [
                _df[_df[class_col] == c]["correct"].mean() for c in _classes
            ]
        return pd.DataFrame(_rows, index=[_fmt(c) for c in _classes]).T

    _agree = _pairwise_agreement(_names)
    _corr_mat, _lbls = _correctness_matrix(_names)
    _class_acc = _class_accuracy(_names, _class_col)
    _n = _corr_mat.shape[0]
    _n_hard = int((_corr_mat.sum(axis=1) == 0).sum())

    _fig_agree, (_ax0, _ax1, _ax2) = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(
        _agree,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.5,
        square=True,
        ax=_ax0,
        cbar_kws={"label": "% same prediction"},
    )
    _ax0.set_title(
        f"{_task} — Pairwise Prediction Agreement (n={_n})", fontweight="bold"
    )

    _im = _ax1.imshow(
        _corr_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest"
    )
    _ax1.set_xticks(range(len(_lbls)))
    _ax1.set_xticklabels(_lbls)
    _ax1.set_yticks([])
    _ax1.set_ylabel("Sample (easiest → hardest)")
    _ax1.set_title(f"{_task} — Sample Correctness", fontweight="bold")
    _ax1.text(
        0.01,
        0.99,
        f"All wrong: {_n_hard} samples",
        transform=_ax1.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )
    plt.colorbar(_im, ax=_ax1, fraction=0.046, label="Correct")

    sns.heatmap(
        _class_acc,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=_ax2,
        cbar_kws={"label": "Accuracy"},
    )
    _class_label = "BI-RADS class" if _task == "Cancer" else "Density class"
    _ax2.set_xlabel(_class_label)
    _ax2.set_ylabel("Model")
    _ax2.set_title(f"{_task} — Per-Class Accuracy", fontweight="bold")

    _fig_agree.tight_layout()
    _fig_agree
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Grad‑CAM Heatmap Overview

    The panel below shows **Grad‑CAM** attention maps for the selected breast‑image from the annotated part of test dataset, comparing two user‑chosen models.
    Adjust the controls above to switch models, swap behaviour to visualize Grad-CAMs generated from tru labels (default is from predicted), or filter to only correctly classified cases.
    """)
    return


@app.cell(hide_code=True)
def _(
    cancer_ajive_model,
    density_ajive_model,
    mo,
    model_cancer,
    model_density,
):
    model_options = {
        "cancer original": (model_cancer, None),
        "density original": (model_density, None),
    }
    model_key_map = {
        "cancer original": "original_cancer",
        "density original": "original_density",
    }

    for head in head_names(cancer_ajive_model):
        model_options[f"cancer ajive {head}"] = (cancer_ajive_model, head)
        model_key_map[f"cancer ajive {head}"] = f"cancer_ajive_{head}"
    for head in head_names(density_ajive_model):
        model_options[f"density ajive {head}"] = (density_ajive_model, head)
        model_key_map[f"density ajive {head}"] = f"density_ajive_{head}"

    model_keys = list(model_options.keys())

    def _model_default(idx):
        return (
            model_keys[idx]
            if idx < len(model_keys)
            else (model_keys[-1] if model_keys else None)
        )

    preview_model_1 = mo.ui.dropdown(
        options=model_keys, value=_model_default(0), label="Model 1"
    )
    preview_model_2 = mo.ui.dropdown(
        options=model_keys, value=_model_default(1), label="Model 2"
    )
    preview_model_3 = mo.ui.dropdown(
        options=model_keys, value=_model_default(2), label="Model 3"
    )
    preview_model_4 = mo.ui.dropdown(
        options=model_keys, value=_model_default(3), label="Model 4"
    )

    use_true_class_preview = mo.ui.checkbox(
        label="Use true class for Grad-CAM",
        value=False,
    )
    limit_correct_1 = mo.ui.checkbox(label="Correct by Model 1", value=False)
    limit_correct_2 = mo.ui.checkbox(label="Correct by Model 2", value=False)
    limit_correct_3 = mo.ui.checkbox(label="Correct by Model 3", value=False)
    limit_correct_4 = mo.ui.checkbox(label="Correct by Model 4", value=False)

    preview_ui = mo.vstack(
        [
            mo.hstack(
                [preview_model_1, preview_model_2, preview_model_3, preview_model_4]
            ),
            mo.hstack(
                [
                    use_true_class_preview,
                    limit_correct_1,
                    limit_correct_2,
                    limit_correct_3,
                    limit_correct_4,
                ]
            ),
        ]
    )
    return (
        limit_correct_1,
        limit_correct_2,
        limit_correct_3,
        limit_correct_4,
        model_key_map,
        model_keys,
        model_options,
        preview_model_1,
        preview_model_2,
        preview_model_3,
        preview_model_4,
        preview_ui,
        use_true_class_preview,
    )


@app.cell(hide_code=True)
def _(
    annot_test,
    limit_correct_1,
    limit_correct_2,
    limit_correct_3,
    limit_correct_4,
    mo,
    model_key_map,
    model_options,
    np,
    preview_model_1,
    preview_model_2,
    preview_model_3,
    preview_model_4,
    results_by_model_pred,
    results_by_model_true,
    use_true_class_preview,
):
    preview_message = None
    if annot_test.empty:
        preview_indices = np.array([], dtype=int)
        preview_message = "No annotations with findings were found in the test set."
    elif not model_options:
        preview_indices = np.array([], dtype=int)
        preview_message = "No models available for Grad-CAM preview."
    else:
        active_results = (
            results_by_model_true
            if use_true_class_preview.value
            else results_by_model_pred
        )

        _filter_pairs = [
            (limit_correct_1.value, preview_model_1.value),
            (limit_correct_2.value, preview_model_2.value),
            (limit_correct_3.value, preview_model_3.value),
            (limit_correct_4.value, preview_model_4.value),
        ]

        _combined_mask = np.ones(len(annot_test), dtype=bool)
        _any_filter = False
        for _checked, _model_label in _filter_pairs:
            if _checked and _model_label is not None:
                _mkey = model_key_map.get(_model_label)
                if _mkey in active_results:
                    _combined_mask &= (
                        active_results[_mkey]["correct"].to_numpy().astype(bool)
                    )
                    _any_filter = True

        preview_indices = (
            np.flatnonzero(_combined_mask)
            if _any_filter
            else np.arange(len(annot_test))
        )

    preview_pos = mo.ui.slider(
        start=0,
        stop=max(len(preview_indices) - 1, 0),
        step=1,
        value=min(10, max(len(preview_indices) - 1, 0)),
        label="Annotation index",
        full_width=True,
    )

    preview_status = (
        mo.md(preview_message)
        if preview_message is not None
        else mo.md(f"Available annotations: {len(preview_indices)}")
    )
    return preview_indices, preview_pos, preview_status


@app.cell(hide_code=True)
def _(
    Rectangle,
    annot_test,
    compute_gradcam_data,
    dataloader,
    mo,
    model_options,
    np,
    pd,
    plt,
    preview_indices,
    preview_model_1,
    preview_model_2,
    preview_model_3,
    preview_model_4,
    preview_pos,
    torch,
    use_true_class_preview,
):
    _preview_models = [
        preview_model_1.value,
        preview_model_2.value,
        preview_model_3.value,
        preview_model_4.value,
    ]

    if len(preview_indices) == 0 or all(v is None for v in _preview_models):
        preview_output = mo.md("No annotations match the current filter.")
    else:
        selected_idx = int(preview_indices[int(preview_pos.value)])
        annot_row = annot_test.iloc[selected_idx]

        _image_idx = np.where(
            dataloader.test_dataset.image_ids == annot_row["image_id"]
        )[0][0]
        _image_tensor, _label1_t, _label2_t = dataloader.test_dataset[_image_idx]
        _height, _width = _image_tensor.shape[1], _image_tensor.shape[2]
        _y1_true = int(_label1_t.item() if torch.is_tensor(_label1_t) else _label1_t)
        _y2_true = int(_label2_t.item() if torch.is_tensor(_label2_t) else _label2_t)

        _image_np = _image_tensor.detach().permute(1, 2, 0).cpu().numpy()
        _image_np = np.clip(_image_np * 69.6431 + 108.3328, 0, 255) / 255.0

        _image_rows = annot_test[annot_test["image_id"] == annot_row["image_id"]]

        def _bbox_pixels(row):
            x_min = int(row["resized_xmin"] * _width / 912)
            y_min = int(row["resized_ymin"] * _height / 1520)
            x_max = int(row["resized_xmax"] * _width / 912)
            y_max = int(row["resized_ymax"] * _height / 1520)
            return (
                max(0, min(x_min, _width - 1)),
                max(0, min(y_min, _height - 1)),
                max(0, min(x_max, _width)),
                max(0, min(y_max, _height)),
            )

        def _draw_bboxes(ax):
            for _, row in _image_rows.iterrows():
                x_min, y_min, x_max, y_max = _bbox_pixels(row)
                color = "cyan" if row.name == selected_idx else "yellow"
                style = "-" if row.name == selected_idx else "--"
                ax.add_patch(
                    Rectangle(
                        (x_min, y_min),
                        x_max - x_min,
                        y_max - y_min,
                        linewidth=2,
                        edgecolor=color,
                        facecolor="none",
                        linestyle=style,
                    )
                )

        _preview_fig, _preview_axes = plt.subplots(2, 3, figsize=(18, 12))

        _preview_axes[0, 0].imshow(_image_np, cmap="gray")
        _preview_axes[0, 0].set_title(
            f"Original Image\nTrue: BI-RADS {_y1_true + 1}, Density {_y2_true}"
        )
        _preview_axes[0, 0].axis("off")
        _draw_bboxes(_preview_axes[0, 0])

        _overlay_positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
        _all_metrics = []

        for _pos, _model_label in zip(_overlay_positions, _preview_models):
            _ax = _preview_axes[_pos]
            if _model_label is None:
                _ax.axis("off")
                continue
            _model, _head_name = model_options[_model_label]
            _img_np, _cam, _pred_class, _enrichment, _, _ = compute_gradcam_data(
                _model,
                annot_row,
                dataloader.test_dataset,
                head_name=_head_name,
                use_true_class=use_true_class_preview.value,
            )
            _ax.imshow(_img_np, cmap="gray")
            _im = _ax.imshow(_cam, cmap="jet", alpha=0.5)
            _ax.set_title(
                f"{_model_label}\nPred: {_pred_class + 1} | Enrich: {_enrichment:.2f}×"
            )
            _ax.axis("off")
            _draw_bboxes(_ax)
            plt.colorbar(_im, ax=_ax, fraction=0.046)
            _all_metrics.append(
                {
                    "model": _model_label,
                    "predicted_class": _pred_class,
                    "enrichment": round(_enrichment, 3),
                }
            )

        _preview_axes[1, 0].axis("off")

        _preview_fig.suptitle(
            f"Grad-CAM Comparison | {annot_row['finding_category']}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        _metrics_df = pd.DataFrame(_all_metrics)
        _info = mo.md(f"Selected annotation index: {selected_idx}")
        preview_output = mo.vstack([_info, _preview_fig, _metrics_df])
    return (preview_output,)


@app.cell
def _(mo, preview_output, preview_pos, preview_status, preview_ui):
    mo.vstack([preview_ui, preview_status, preview_pos, preview_output])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Classification accuracy comparison
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    metric_select = mo.ui.dropdown(
        options=["Accuracy", "F1-Score", "Balanced Accuracy"],
        value="F1-Score",
        label="Metric to plot",
    )
    metric_select
    return (metric_select,)


@app.cell(hide_code=True)
def _(
    accuracy_score,
    balanced_accuracy_score,
    cache_dir,
    cancer_ajive_model,
    dataloader,
    density_ajive_model,
    device,
    f1_score,
    metric_select,
    mo,
    model_cancer,
    model_density,
    np,
    os,
    pd,
    plt,
    sns,
    torch,
    tqdm,
):
    results_path = os.path.join(cache_dir, "classification_results_df.pkl")

    if os.path.exists(results_path):
        results_df = pd.read_pickle(results_path)
    else:
        test_dataloader = dataloader.test_dataloader()

        all_predictions = {
            "original_cancer": [],
            "original_density": [],
            "cancer_joint": [],
            "cancer_individual": [],
            "cancer_both": [],
            "density_joint": [],
            "density_individual": [],
            "density_both": [],
        }
        all_labels_cancer = []
        all_labels_density = []

        _head_names = (
            list(cancer_ajive_model.head_names)
            if hasattr(cancer_ajive_model, "head_names")
            else list(cancer_ajive_model.heads.keys())
        )

        with torch.no_grad():
            for batch in tqdm.tqdm(test_dataloader, desc="Evaluating batches"):
                images, cancer_labels, density_labels = batch
                images = images.to(device)

                cancer_pred = model_cancer(images).argmax(dim=1).cpu().numpy()
                density_pred = model_density(images).argmax(dim=1).cpu().numpy()

                all_predictions["original_cancer"].extend(cancer_pred)
                all_predictions["original_density"].extend(density_pred)

                for _head_name in _head_names:
                    cancer_ajive_model.set_active_head(_head_name)
                    head_pred = (
                        cancer_ajive_model(images, head_name=_head_name)
                        .argmax(dim=1)
                        .cpu()
                        .numpy()
                    )
                    all_predictions[f"cancer_{_head_name}"].extend(head_pred)

                for _head_name in _head_names:
                    density_ajive_model.set_active_head(_head_name)
                    head_pred = (
                        density_ajive_model(images, head_name=_head_name)
                        .argmax(dim=1)
                        .cpu()
                        .numpy()
                    )
                    all_predictions[f"density_{_head_name}"].extend(head_pred)

                all_labels_cancer.extend(cancer_labels.cpu().numpy())
                all_labels_density.extend(density_labels.cpu().numpy())

        all_labels_cancer = np.array(all_labels_cancer)
        all_labels_density = np.array(all_labels_density)

        results = []
        for _model_name in [
            "original_cancer",
            "cancer_joint",
            "cancer_individual",
            "cancer_both",
        ]:
            preds = np.array(all_predictions[_model_name])
            acc = accuracy_score(all_labels_cancer, preds)
            bal_acc = balanced_accuracy_score(all_labels_cancer, preds)
            f1 = f1_score(all_labels_cancer, preds, average="macro", zero_division=0)
            results.append(
                {
                    "Model": _model_name.replace("_", " ").title(),
                    "Task": "Cancer",
                    "Accuracy": acc,
                    "Balanced Accuracy": bal_acc,
                    "F1-Score": f1,
                }
            )

        for _model_name in [
            "original_density",
            "density_joint",
            "density_individual",
            "density_both",
        ]:
            preds = np.array(all_predictions[_model_name])
            acc = accuracy_score(all_labels_density, preds)
            bal_acc = balanced_accuracy_score(all_labels_density, preds)
            f1 = f1_score(all_labels_density, preds, average="macro", zero_division=0)
            results.append(
                {
                    "Model": _model_name.replace("_", " ").title(),
                    "Task": "Density",
                    "Accuracy": acc,
                    "Balanced Accuracy": bal_acc,
                    "F1-Score": f1,
                }
            )

        results_df = pd.DataFrame(results)
        results_df.to_pickle(results_path)

    metric = metric_select.value

    _fig, _ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        data=results_df,
        x="Model",
        y=metric,
        hue="Task",
        ax=_ax,
    )
    _ax.set_title(f"Classification performance ({metric})")
    _ax.set_ylabel(metric)
    _ax.set_xlabel("Model")
    _ax.tick_params(axis="x", rotation=45)
    _ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    mo.vstack([results_df, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Grad-CAM localization metrics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    comparison_set = mo.ui.dropdown(
        options=[
            "Cancer: original + AJIVE heads",
            "Density: original + AJIVE heads",
            "Both: original + AJIVE heads",
            "Cancer original",
        ],
        value="Both: original + AJIVE heads",
        label="Model set",
    )
    use_true_class_loc = mo.ui.checkbox(
        label="Use true class for Grad-CAM",
        value=False,
    )
    correct_only = mo.ui.checkbox(
        label="Plot only correctly classified samples",
        value=True,
    )
    stratify = mo.ui.dropdown(
        options=[
            "None",
            "finding_category",
            "BI-RADS",
            "finding_category -> BI-RADS",
            "BI-RADS -> finding_category",
        ],
        value="None",
        label="Stratify",
    )
    top_k = mo.ui.slider(
        start=0,
        stop=10,
        step=1,
        value=3,
        label="Top K finding categories (0 = all)",
    )

    loc_ui = mo.vstack(
        [
            comparison_set,
            use_true_class_loc,
            correct_only,
            stratify,
            top_k,
        ]
    )
    return (
        comparison_set,
        correct_only,
        loc_ui,
        stratify,
        top_k,
        use_true_class_loc,
    )


@app.cell(hide_code=True)
def _(
    comparison_set,
    correct_only,
    loc_ui,
    mo,
    plot_localization_metrics,
    results_list_pred,
    results_list_true,
    stratify,
    top_k,
    use_true_class_loc,
):
    def _filter_results(results_list, selection):
        if "Both" in selection:
            return results_list
        if "Cancer original" in selection:
            return [results_list[0]]
        keep = []
        for df in results_list:
            model_name = df["model_name"].iloc[0] if not df.empty else ""
            if "Cancer" in selection and "cancer" in model_name:
                keep.append(df)
            if "Density" in selection and "density" in model_name:
                keep.append(df)
        return keep

    selection = comparison_set.value
    results_list = _filter_results(
        results_list_true if use_true_class_loc.value else results_list_pred, selection
    )

    stratify_value = stratify.value
    if stratify_value == "None":
        stratify_cols = None
    elif stratify_value == "finding_category -> BI-RADS":
        stratify_cols = ["finding_category", "BI-RADS"]
    elif stratify_value == "BI-RADS -> finding_category":
        stratify_cols = ["BI-RADS", "finding_category"]
    else:
        stratify_cols = stratify_value

    top_k_value = int(top_k.value) if int(top_k.value) > 0 else None

    if results_list is not None and len(results_list) > 0:
        cancer_list = [
            df for df in results_list if "cancer" in df["model_name"].iloc[0].lower()
        ]
        density_list = [
            df for df in results_list if "density" in df["model_name"].iloc[0].lower()
        ]
        results_list = cancer_list + density_list
        method_labels = [f"{df['model_name'].iloc[0]}" for df in results_list]
        fig, _ = plot_localization_metrics(
            results_list,
            method_labels=method_labels,
            stratify=stratify_cols,
            correct_only=correct_only.value,
            top_k=top_k_value,
        )
        box_res = fig
    else:
        box_res = mo.md("No localization results to plot.")
    mo.vstack([loc_ui, box_res])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## GradCAM Similarity Analysis

    This section measures how **similarly two models attend to image regions** via their
    Grad-CAM heatmaps, and uses that agreement as a proxy for prediction confidence.
    The core idea: when two models highlight the same areas, we trust the prediction more;
    when they disagree, the case is ambiguous and we may want to abstain.

    ### Metrics

    Three pixel-level similarity metrics are computed between the two heatmaps:

    | Metric | What it measures |
    |---|---|
    | **Pearson** | Linear correlation of flattened heatmap values — sensitive to global activation pattern similarity |
    | **SSIM** | Global structural similarity (luminance + contrast + structure) via numpy |
    | **IoU top-10%** | Intersection-over-Union of the top-10% activated pixels — focuses on the most salient regions |

    ### How to use

    1. **Model A / Model B** — choose any two models to compare (e.g. original cancer vs AJIVE individual head).
       Results are cached per model pair; changing the selection recomputes and saves a new cache file.
    2. **Similarity metric** — selects which of the three metrics drives the threshold filter.
    3. **Classification task** — whether to evaluate Cancer (BI-RADS) or Density classification performance.
    4. **Predictions from** — which model's predictions are used to compute the classification metric.
    5. **Classification metric** — F1 macro / weighted, Accuracy, or Balanced Accuracy.
       F1 variants use `labels = sim_df[label_col].unique()` so the label universe stays fixed
       as samples are filtered out, giving a fair macro average across thresholds.
    """)
    return


@app.cell(hide_code=True)
def _(mo, model_keys):
    sim_model_1 = mo.ui.dropdown(
        options=model_keys, value=model_keys[0], label="Model A"
    )
    sim_model_2 = mo.ui.dropdown(
        options=model_keys,
        value=model_keys[3] if len(model_keys) > 3 else model_keys[-1],
        label="Model B",
    )
    sim_target_ui = mo.ui.dropdown(
        options=["Predicted (each)", "True class", "Predicted by Model A"],
        value="Predicted (each)",
        label="Target class for Grad-CAM",
    )
    controls_1 = mo.vstack([mo.hstack([sim_model_1, sim_model_2, sim_target_ui])])
    return controls_1, sim_model_1, sim_model_2, sim_target_ui


@app.cell(hide_code=True)
def _(
    cache_dir,
    dataloader,
    mo,
    model_options,
    np,
    os,
    pd,
    sim_model_1,
    sim_model_2,
    torch,
    tqdm,
):
    def _pearson(c1, c2):
        flat1, flat2 = c1.ravel(), c2.ravel()
        return float(np.corrcoef(flat1, flat2)[0, 1])

    def _iou_topk(c1, c2, k=0.10):
        t1, t2 = np.quantile(c1, 1 - k), np.quantile(c2, 1 - k)
        m1, m2 = c1 >= t1, c2 >= t2
        union = (m1 | m2).sum()
        return float((m1 & m2).sum() / union) if union > 0 else 0.0

    def _ssim(c1, c2, C1=1e-4, C2=9e-4):
        mu1, mu2 = c1.mean(), c2.mean()
        s12 = ((c1 - mu1) * (c2 - mu2)).mean()
        denom = (mu1**2 + mu2**2 + C1) * (c1.std() ** 2 + c2.std() ** 2 + C2)
        return (
            float(((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / denom)
            if denom != 0
            else 0.0
        )

    def _gcam_raw(model, img_tensor, head_name, target):
        dev = next(model.parameters()).device
        tc = torch.tensor([target], device=dev) if target is not None else None
        _i = img_tensor.unsqueeze(0).to(dev)
        if head_name is None or not hasattr(model, "head_names"):
            cam, pred = model.generate_grad_cam(_i, target_class=tc)
        else:
            cam, pred = model.generate_grad_cam(
                _i, target_class=tc, head_name=head_name
            )
        return cam.detach().cpu(), pred.item()

    def _resize(cam_t, H, W):
        return (
            torch.nn.functional.interpolate(
                cam_t.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            )
            .squeeze(1)[0]
            .numpy()
        )

    def _sim_metrics(ca, cb, suffix):
        return {
            f"pearson_{suffix}": _pearson(ca, cb),
            f"ssim_{suffix}": _ssim(ca, cb),
            f"iou_topk_{suffix}": _iou_topk(ca, cb),
        }

    _m1_key = sim_model_1.value.replace(" ", "_")
    _m2_key = sim_model_2.value.replace(" ", "_")
    _sim_cache_path = os.path.join(
        cache_dir, f"sim_{_m1_key}_vs_{_m2_key}_all_full.pkl"
    )

    if os.path.exists(_sim_cache_path):
        sim_df = pd.read_pickle(_sim_cache_path)
    else:
        _m1, _h1 = model_options[sim_model_1.value]
        _m2, _h2 = model_options[sim_model_2.value]
        _dataset = dataloader.test_dataset
        _rows = []
        for _didx in tqdm.tqdm(
            range(len(_dataset)), desc="Computing GradCAM similarities"
        ):
            _image_tensor, _label1, _label2 = _dataset[_didx]
            _iid = _dataset.image_ids[_didx]
            _y1 = int(_label1.item() if torch.is_tensor(_label1) else _label1)
            _y2 = int(_label2.item() if torch.is_tensor(_label2) else _label2)
            _true1 = _y1 if getattr(_m1, "task", 1) == 1 else _y2
            _true2 = _y1 if getattr(_m2, "task", 1) == 1 else _y2
            H, W = _image_tensor.shape[1], _image_tensor.shape[2]
            try:
                _c1_pred_t, _pred1 = _gcam_raw(_m1, _image_tensor, _h1, None)
                _c2_pred2_t, _pred2 = _gcam_raw(_m2, _image_tensor, _h2, None)
                _c2_pred1_t = (
                    _c2_pred2_t
                    if _pred1 == _pred2
                    else _gcam_raw(_m2, _image_tensor, _h2, _pred1)[0]
                )
                _c1_true_t = (
                    _c1_pred_t
                    if _true1 == _pred1
                    else _gcam_raw(_m1, _image_tensor, _h1, _true1)[0]
                )
                _c2_true_t = (
                    _c2_pred2_t
                    if _true2 == _pred2
                    else _gcam_raw(_m2, _image_tensor, _h2, _true2)[0]
                )
            except Exception:
                continue
            _r1p = _resize(_c1_pred_t, H, W)
            _r2p2 = _resize(_c2_pred2_t, H, W)
            _r2p1 = (
                _r2p2 if (_c2_pred1_t is _c2_pred2_t) else _resize(_c2_pred1_t, H, W)
            )
            _r1t = _r1p if (_c1_true_t is _c1_pred_t) else _resize(_c1_true_t, H, W)
            _r2t = _r2p2 if (_c2_true_t is _c2_pred2_t) else _resize(_c2_true_t, H, W)
            _row = {
                "image_id": _iid,
                "y1_label": _y1,
                "y2_label": _y2,
                "pred_1": _pred1,
                "pred_2": _pred2,
            }
            _row.update(_sim_metrics(_r1p, _r2p2, "pred_each"))
            _row.update(_sim_metrics(_r1t, _r2t, "true"))
            _row.update(_sim_metrics(_r1p, _r2p1, "pred_a"))
            _rows.append(_row)
        sim_df = pd.DataFrame(_rows)
        pd.to_pickle(sim_df, _sim_cache_path)
    mo.md(
        f"Similarity computed for **{len(sim_df)}** images — variants: `pred_each`, `true`, `pred_a`."
    )
    return (sim_df,)


@app.cell(hide_code=True)
def _(mo):
    sim_metric_ui = mo.ui.dropdown(
        options=["pearson", "ssim", "iou_topk"],
        value="pearson",
        label="Similarity metric",
    )
    sim_task_ui = mo.ui.dropdown(
        options=["Cancer (BI-RADS)", "Density"],
        value="Cancer (BI-RADS)",
        label="Classification task",
    )
    sim_pred_ui = mo.ui.dropdown(
        options=["Model A", "Model B"], value="Model A", label="Predictions from"
    )
    clf_metric_ui = mo.ui.dropdown(
        options=["F1 macro", "F1 weighted", "Accuracy", "Balanced Accuracy"],
        value="F1 macro",
        label="Classification metric",
    )
    controls_2 = mo.vstack(
        [mo.hstack([sim_metric_ui, sim_task_ui, sim_pred_ui]), clf_metric_ui]
    )
    return clf_metric_ui, controls_2, sim_metric_ui, sim_pred_ui, sim_task_ui


@app.cell(hide_code=True)
def _(
    accuracy_score,
    balanced_accuracy_score,
    clf_metric_ui,
    controls_1,
    controls_2,
    f1_score,
    mo,
    np,
    plt,
    sim_df,
    sim_metric_ui,
    sim_model_1,
    sim_model_2,
    sim_pred_ui,
    sim_target_ui,
    sim_task_ui,
):
    _target_suffix = {
        "Predicted (each)": "pred_each",
        "True class": "true",
        "Predicted by Model A": "pred_a",
    }[sim_target_ui.value]

    _sim_metric = sim_metric_ui.value
    _sim_col = f"{_sim_metric}_{_target_suffix}"
    _label_col = "y1_label" if "Cancer" in sim_task_ui.value else "y2_label"
    _pred_col = "pred_1" if sim_pred_ui.value == "Model A" else "pred_2"

    _all_labels = np.sort(sim_df[_label_col].unique())
    _score_fns = {
        "F1 macro": lambda y, p: f1_score(
            y, p, average="macro", zero_division=0, labels=_all_labels
        ),
        "F1 weighted": lambda y, p: f1_score(
            y, p, average="weighted", zero_division=0, labels=_all_labels
        ),
        "Accuracy": lambda y, p: accuracy_score(y, p),
        "Balanced Accuracy": lambda y, p: balanced_accuracy_score(y, p),
    }
    _score_fn = _score_fns[clf_metric_ui.value]
    _clf_label = clf_metric_ui.value

    _vals = sim_df[_sim_col].values
    _thresholds = np.linspace(_vals.min(), _vals.max(), 150)

    _scores, _coverages = [], []
    for _t in _thresholds:
        _mask = sim_df[_sim_col] >= _t
        _sub = sim_df[_mask]
        if len(_sub) < 2:
            continue
        _scores.append(_score_fn(_sub[_label_col], _sub[_pred_col]))
        _coverages.append(_mask.mean())

    _thresholds = _thresholds[: len(_scores)]

    _fig, _ax1 = plt.subplots(figsize=(10, 5))
    _ax2 = _ax1.twinx()

    _ax1.plot(_thresholds, _scores, color="steelblue", linewidth=2, label=_clf_label)
    _ax2.plot(
        _thresholds,
        _coverages,
        color="darkorange",
        linewidth=2,
        linestyle="--",
        label="Coverage",
    )

    _baseline = _score_fn(sim_df[_label_col], sim_df[_pred_col])
    _ax1.axhline(
        _baseline,
        color="steelblue",
        linestyle=":",
        alpha=0.5,
        label=f"Baseline = {_baseline:.3f}",
    )

    _ax1.set_xlabel(f"{_sim_col} threshold", fontsize=12)
    _ax1.set_ylabel(_clf_label, color="steelblue", fontsize=11)
    _ax2.set_ylabel("Coverage (fraction accepted)", color="darkorange", fontsize=11)
    _ax1.tick_params(axis="y", labelcolor="steelblue")
    _ax2.tick_params(axis="y", labelcolor="darkorange")

    _lines1, _labels1 = _ax1.get_legend_handles_labels()
    _lines2, _labels2 = _ax2.get_legend_handles_labels()
    _ax1.legend(_lines1 + _lines2, _labels1 + _labels2, loc="center left")

    _task_label = "Cancer (BI-RADS)" if _label_col == "y1_label" else "Density"
    _pred_label = sim_model_1.value if _pred_col == "pred_1" else sim_model_2.value
    _title_line1 = (
        f"{_clf_label} vs {_sim_col} threshold — {_task_label} | Preds: {_pred_label}"
    )
    _title_line2 = (
        f"target={sim_target_ui.value} | {sim_model_1.value} vs {sim_model_2.value}"
    )
    _ax1.set_title(f"{_title_line1}\n{_title_line2}", fontsize=11)
    _ax1.grid(alpha=0.3)
    plt.tight_layout()
    mo.vstack([controls_1, controls_2, _fig])
    return


@app.cell
def _(sim_df):
    sim_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All-Class Grad-CAM Similarity (Cancer: original vs AJIVE individual)

    Unlike the section above, this analysis does **not** pick a single target class. For the
    fixed pair **cancer original** vs **cancer AJIVE individual**, it generates a Grad-CAM map
    for **every** BI-RADS class on both models and compares the corresponding class-maps,
    yielding **one similarity value per class** for each image (e.g. 5 values for 5 classes).

    The abstention figure then **aggregates** those per-class similarities into a single score —
    choose `mean`, `max`, `min`, `std`, `median`, or `range` — and sweeps a threshold over it to
    trade classification performance against coverage. For agreement-style aggregations
    (mean/max/min/median) higher means the models attend more alike, so accept **above** the
    threshold; for dispersion aggregations (std/range) lower means more agreement, so accept
    **below**. The direction is set with the manual toggle.

    The full-test computation is expensive, so results are cached to disk and reloaded on reruns.
    """)
    return


@app.cell(hide_code=True)
def _(cache_dir, dataloader, mo, model_options, np, os, pd, torch, tqdm):
    def _pearson(c1, c2):
        x = c1.ravel()
        y = c2.ravel()

        sx = np.std(x)
        sy = np.std(y)

        if sx == 0 and sy == 0:
            return 1.0 if np.allclose(x, y) else 0.0

        if sx == 0 or sy == 0:
            return 0.0

        return float(np.corrcoef(x, y)[0, 1])

    def _iou_topk(c1, c2, k=0.10):
        t1, t2 = np.quantile(c1, 1 - k), np.quantile(c2, 1 - k)
        m1, m2 = c1 >= t1, c2 >= t2
        union = (m1 | m2).sum()
        return float((m1 & m2).sum() / union) if union > 0 else 0.0

    def _ssim(c1, c2, C1=1e-4, C2=9e-4):
        mu1, mu2 = c1.mean(), c2.mean()
        s12 = ((c1 - mu1) * (c2 - mu2)).mean()
        denom = (mu1**2 + mu2**2 + C1) * (c1.std() ** 2 + c2.std() ** 2 + C2)
        return (
            float(((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / denom)
            if denom != 0
            else 0.0
        )

    def _gcam_raw(model, img_tensor, head_name, target):
        dev = next(model.parameters()).device
        tc = torch.tensor([target], device=dev) if target is not None else None
        _i = img_tensor.unsqueeze(0).to(dev)
        if head_name is None or not hasattr(model, "head_names"):
            cam, pred = model.generate_grad_cam(_i, target_class=tc)
        else:
            cam, pred = model.generate_grad_cam(
                _i, target_class=tc, head_name=head_name
            )
        return cam.detach().cpu(), pred.item()

    def _resize(cam_t, H, W):
        return (
            torch.nn.functional.interpolate(
                cam_t.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            )
            .squeeze(1)[0]
            .numpy()
        )

    _mA, _hA = model_options["cancer original"]
    _mB, _hB = model_options["cancer ajive individual"]
    _dataset = dataloader.test_dataset

    _probe_img = _dataset[0][0]
    with torch.no_grad():
        _dev_a = next(_mA.parameters()).device
        num_classes = int(_mA(_probe_img.unsqueeze(0).to(_dev_a)).shape[1])

    _allsim_cache_path = os.path.join(
        cache_dir, "sim_allclass_cancer_orig_vs_ajive_indiv.pkl"
    )

    if os.path.exists(_allsim_cache_path):
        allsim_df = pd.read_pickle(_allsim_cache_path)
    else:
        _rows = []
        for _didx in tqdm.tqdm(
            range(len(_dataset)), desc="All-class GradCAM similarities"
        ):
            _image_tensor, _label1, _label2 = _dataset[_didx]
            _iid = _dataset.image_ids[_didx]
            _y1 = int(_label1.item() if torch.is_tensor(_label1) else _label1)
            _y2 = int(_label2.item() if torch.is_tensor(_label2) else _label2)
            _H, _W = _image_tensor.shape[1], _image_tensor.shape[2]
            try:
                _pred1 = None
                _pred2 = None
                _row = {"image_id": _iid, "y1_label": _y1, "y2_label": _y2}
                for _c in range(num_classes):
                    _ca_t, _pa = _gcam_raw(_mA, _image_tensor, _hA, _c)
                    _cb_t, _pb = _gcam_raw(_mB, _image_tensor, _hB, _c)
                    if _pred1 is None:
                        _pred1, _pred2 = _pa, _pb
                    _ra = _resize(_ca_t, _H, _W)
                    _rb = _resize(_cb_t, _H, _W)
                    _row[f"pearson_class_{_c}"] = _pearson(_ra, _rb)
                    _row[f"ssim_class_{_c}"] = _ssim(_ra, _rb)
                    _row[f"iou_topk_class_{_c}"] = _iou_topk(_ra, _rb)
            except Exception:
                continue
            _row["pred_1"] = _pred1
            _row["pred_2"] = _pred2
            _rows.append(_row)
        allsim_df = pd.DataFrame(_rows)
        pd.to_pickle(allsim_df, _allsim_cache_path)

    mo.md(
        f"All-class similarity computed for **{len(allsim_df)}** images "
        f"across **{num_classes}** classes (cancer original vs AJIVE individual)."
    )
    return allsim_df, num_classes


@app.cell(hide_code=True)
def _(mo):
    allsim_metric_ui = mo.ui.dropdown(
        options=["pearson", "ssim", "iou_topk"],
        value="pearson",
        label="Base similarity metric",
    )
    allsim_agg_ui = mo.ui.dropdown(
        options=[
            "mean",
            "max",
            "min",
            "std",
            "median",
            "range",
            "predicted class (Model A)",
            "true class",
        ],
        value="mean",
        label="Aggregate 5 class-similarities by",
    )
    allsim_pred_ui = mo.ui.dropdown(
        options=["Model A (original)", "Model B (AJIVE individual)"],
        value="Model A (original)",
        label="Predictions from",
    )
    allsim_clf_metric_ui = mo.ui.dropdown(
        options=["F1 macro", "F1 weighted", "Accuracy", "Balanced Accuracy"],
        value="F1 macro",
        label="Classification metric",
    )
    allsim_direction_ui = mo.ui.dropdown(
        options=["Accept above threshold", "Accept below threshold"],
        value="Accept above threshold",
        label="Abstention direction",
    )
    allsim_controls = mo.vstack(
        [
            mo.hstack([allsim_metric_ui, allsim_agg_ui, allsim_pred_ui]),
            mo.hstack([allsim_clf_metric_ui, allsim_direction_ui]),
        ]
    )
    return (
        allsim_agg_ui,
        allsim_clf_metric_ui,
        allsim_controls,
        allsim_direction_ui,
        allsim_metric_ui,
        allsim_pred_ui,
    )


@app.cell(hide_code=True)
def _(
    accuracy_score,
    allsim_agg_ui,
    allsim_clf_metric_ui,
    allsim_controls,
    allsim_df,
    allsim_direction_ui,
    allsim_metric_ui,
    allsim_pred_ui,
    balanced_accuracy_score,
    f1_score,
    mo,
    np,
    num_classes,
    plt,
):
    _metric = allsim_metric_ui.value
    _agg = allsim_agg_ui.value
    _pred_col = "pred_1" if allsim_pred_ui.value == "Model A (original)" else "pred_2"
    _label_col = "y1_label"

    _class_cols = [f"{_metric}_class_{_c}" for _c in range(num_classes)]
    _per_class = allsim_df[_class_cols].to_numpy()
    _rows_idx = np.arange(len(_per_class))

    _agg_fns = {
        "mean": lambda a: a.mean(axis=1),
        "max": lambda a: a.max(axis=1),
        "min": lambda a: a.min(axis=1),
        "std": lambda a: a.std(axis=1),
        "median": lambda a: np.median(a, axis=1),
        "range": lambda a: a.max(axis=1) - a.min(axis=1),
        # Pick the single class-map similarity that matches a reference class per image,
        # replicating the previous section's "pred_a" / "true" target variants.
        "predicted class (Model A)": lambda a: a[
            _rows_idx, allsim_df["pred_1"].to_numpy().clip(0, num_classes - 1)
        ],
        "true class": lambda a: a[
            _rows_idx, allsim_df["y1_label"].to_numpy().clip(0, num_classes - 1)
        ],
    }
    _agg_score = _agg_fns[_agg](_per_class)

    _all_labels = np.sort(allsim_df[_label_col].unique())
    _score_fns = {
        "F1 macro": lambda y, p: f1_score(
            y, p, average="macro", zero_division=0, labels=_all_labels
        ),
        "F1 weighted": lambda y, p: f1_score(
            y, p, average="weighted", zero_division=0, labels=_all_labels
        ),
        "Accuracy": lambda y, p: accuracy_score(y, p),
        "Balanced Accuracy": lambda y, p: balanced_accuracy_score(y, p),
    }
    _score_fn = _score_fns[allsim_clf_metric_ui.value]
    _clf_label = allsim_clf_metric_ui.value

    _accept_above = allsim_direction_ui.value == "Accept above threshold"
    _y_true = allsim_df[_label_col].to_numpy()
    _y_pred = allsim_df[_pred_col].to_numpy()

    _thresholds = np.linspace(_agg_score.min(), _agg_score.max(), 150)
    _scores, _coverages = [], []
    for _t in _thresholds:
        _mask = _agg_score >= _t if _accept_above else _agg_score <= _t
        if _mask.sum() < 2:
            continue
        _scores.append(_score_fn(_y_true[_mask], _y_pred[_mask]))
        _coverages.append(_mask.mean())
    _thresholds = _thresholds[: len(_scores)]

    _fig, _ax1 = plt.subplots(figsize=(10, 5))
    _ax2 = _ax1.twinx()

    _ax1.plot(_thresholds, _scores, color="steelblue", linewidth=2, label=_clf_label)
    _ax2.plot(
        _thresholds,
        _coverages,
        color="darkorange",
        linewidth=2,
        linestyle="--",
        label="Coverage",
    )

    _baseline = _score_fn(_y_true, _y_pred)
    _ax1.axhline(
        _baseline,
        color="steelblue",
        linestyle=":",
        alpha=0.5,
        label=f"Baseline = {_baseline:.3f}",
    )

    _direction_word = "≥" if _accept_above else "≤"
    _ax1.set_xlabel(
        f"{_agg}({_metric}) threshold (accept {_direction_word} t)", fontsize=12
    )
    _ax1.set_ylabel(_clf_label, color="steelblue", fontsize=11)
    _ax2.set_ylabel("Coverage (fraction accepted)", color="darkorange", fontsize=11)
    _ax1.tick_params(axis="y", labelcolor="steelblue")
    _ax2.tick_params(axis="y", labelcolor="darkorange")

    _lines1, _labels1 = _ax1.get_legend_handles_labels()
    _lines2, _labels2 = _ax2.get_legend_handles_labels()
    _ax1.legend(_lines1 + _lines2, _labels1 + _labels2, loc="lower left")

    _pred_label = (
        "cancer original" if _pred_col == "pred_1" else "cancer ajive individual"
    )
    _title_line1 = f"{_clf_label} vs {_agg}({_metric}) threshold — Cancer (BI-RADS) | Preds: {_pred_label}"
    _title_line2 = "all-class similarity | cancer original vs cancer ajive individual"
    _ax1.set_title(f"{_title_line1}\n{_title_line2}", fontsize=11)
    _ax1.grid(alpha=0.3)
    plt.tight_layout()
    mo.vstack([allsim_controls, _fig])
    return


@app.cell
def _(allsim_df):
    allsim_df
    return


if __name__ == "__main__":
    app.run()
