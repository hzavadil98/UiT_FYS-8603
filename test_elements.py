# /// script
# requires-python = "~=3.11"
# dependencies = [
#     "marimo>=0.23.8",
#     "matplotlib==3.10.9",
#     "numpy==2.4.6",
#     "pandas==3.0.3",
#     "pytorch-lightning==2.6.5",
#     "scikit-learn==1.8.0",
#     "seaborn==0.13.2",
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
    # Elements AJIVE Analysis

    Interactive visualisation of shape and texture classifiers — original ResNet backbone and three-headed AJIVE variants — on the synthetic Elements dataset.
    Switch between the **5-element** (training distribution) and **9-element** (OOD) test sets with the dropdown below.
    """)
    return


@app.cell
def _():
    import os
    import sys

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    import tqdm
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    from src import (
        Single_Element_Dataloader,
        Single_view_AJIVE_heads,
        Single_view_model,
    )

    sys.path.insert(0, os.path.abspath("python_packages"))
    from python_packages import AJIVE

    return (
        AJIVE,
        Single_Element_Dataloader,
        Single_view_AJIVE_heads,
        Single_view_model,
        accuracy_score,
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup
    """)
    return


@app.cell
def _(mo, torch):
    _dev_default = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    shape_model_ckpt = mo.ui.text(
        value="artifacts/model-908ij4rg:v0/model.ckpt",
        label="Shape model checkpoint",
        full_width=True,
    )
    texture_model_ckpt = mo.ui.text(
        value="artifacts/model-qtzv9eop:v0/model.ckpt",
        label="Texture model checkpoint",
        full_width=True,
    )
    shape_heads_pt = mo.ui.text(
        value="artifacts/shape_all_heads.pt",
        label="Shape AJIVE heads (.pt)",
        full_width=True,
    )
    texture_heads_pt = mo.ui.text(
        value="artifacts/texture_all_heads.pt",
        label="Texture AJIVE heads (.pt)",
        full_width=True,
    )
    device_select = mo.ui.dropdown(
        options=["mps", "cuda", "cpu"],
        value=_dev_default,
        label="Device",
    )

    mo.vstack(
        [
            shape_model_ckpt,
            texture_model_ckpt,
            shape_heads_pt,
            texture_heads_pt,
            device_select,
        ]
    )
    return (
        device_select,
        shape_heads_pt,
        shape_model_ckpt,
        texture_heads_pt,
        texture_model_ckpt,
    )


@app.cell(hide_code=True)
def _(Single_view_AJIVE_heads, torch):
    def load_all_heads(path, base_model, ajive_model, num_class):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = Single_view_AJIVE_heads(
            model=base_model,
            ajive_model=ajive_model,
            num_class=num_class,
            active_head=checkpoint["active_head"],
            hidden_dim=56,
            drop=0.3,
            learning_rate=1e-3,
        )
        model.load_state_dict(checkpoint["state_dict"])
        return model.eval()

    return (load_all_heads,)


@app.cell
def _(
    AJIVE,
    Single_view_model,
    device_select,
    load_all_heads,
    mo,
    os,
    shape_heads_pt,
    shape_model_ckpt,
    texture_heads_pt,
    texture_model_ckpt,
    torch,
):
    cache_dir = os.path.join("saved_features", "Elements_AJIVE")
    device = device_select.value

    model_shape = Single_view_model.load_from_checkpoint(
        shape_model_ckpt.value, map_location="cpu"
    )
    model_texture = Single_view_model.load_from_checkpoint(
        texture_model_ckpt.value, map_location="cpu"
    )

    _train_cache = torch.load(
        os.path.join(cache_dir, "all_data_train.pt"),
        map_location="cpu",
        weights_only=False,
    )
    aj = AJIVE(init_signal_ranks=(14, 14), n_jobs=4, center=True)
    aj.fit([_train_cache["shape_features"], _train_cache["texture_features"]])

    shape_ajive_model = load_all_heads(
        shape_heads_pt.value, model_shape, aj, num_class=4
    )
    texture_ajive_model = load_all_heads(
        texture_heads_pt.value, model_texture, aj, num_class=3
    )

    model_shape = model_shape.to(device).eval()
    model_texture = model_texture.to(device).eval()
    shape_ajive_model = shape_ajive_model.to(device).eval()
    texture_ajive_model = texture_ajive_model.to(device).eval()

    mo.md(f"Models loaded. AJIVE summary: `{aj.summary()}`. Running on `{device}`.")
    return (
        cache_dir,
        device,
        model_shape,
        model_texture,
        shape_ajive_model,
        texture_ajive_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset Selection
    """)
    return


@app.cell
def _(mo):
    dataset_select = mo.ui.dropdown(
        options=["5-element (train dist)", "9-element (OOD)"],
        value="5-element (train dist)",
        label="Test dataset",
    )
    dataset_select
    return (dataset_select,)


@app.cell(hide_code=True)
def _(Single_Element_Dataloader, dataset_select, mo):
    _common = dict(
        batch_size=64,
        num_workers=4,
        n_train=5000,
        n_val=1000,
        n_test=1000,
        img_size=224,
        allowed_shapes=["square", "circle", "triangle", "plus"],
        allowed_colors=["red", "green", "blue"],
        allowed_textures=["solid", "spots_polka", "stripes_diagonal"],
        element_seed=42,
        loc_seed=123,
        max_resample_attempt_factor=50,
        train_transform=None,
        transform=None,
        use_train_sampler=False,
        sampler_target="shape",
    )

    if dataset_select.value == "5-element (train dist)":
        selected_dataloader = Single_Element_Dataloader(
            element_n=5, element_size=48, element_size_delta=16, **_common
        )
        dataset_key = "5el"
    else:
        selected_dataloader = Single_Element_Dataloader(
            element_n=9, element_size=24, element_size_delta=8, **_common
        )
        dataset_key = "9el"

    _ds = selected_dataloader.test_dataset
    mo.md(
        f"**{dataset_select.value}** — test split: {len(_ds)} samples  \n"
        f"Shapes: `{_ds.allowed_shapes}`  \nTextures: `{_ds.allowed_textures}`"
    )
    return dataset_key, selected_dataloader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Classification Performance
    """)
    return


@app.cell(hide_code=True)
def _(
    accuracy_score,
    balanced_accuracy_score,
    cache_dir,
    dataset_key,
    device,
    f1_score,
    mo,
    model_shape,
    model_texture,
    os,
    pd,
    plt,
    selected_dataloader,
    shape_ajive_model,
    texture_ajive_model,
    torch,
):
    _clf_cache = os.path.join(cache_dir, f"clf_results_{dataset_key}.pkl")

    if os.path.exists(_clf_cache):
        results_df = pd.read_pickle(_clf_cache)
    else:

        def _eval(model, loader, model_name, head_name=None):
            model.eval()
            _preds, _targets = [], []
            with torch.no_grad():
                for _imgs, _sy, _ty in loader:
                    _imgs = _imgs.to(device)
                    if head_name is not None:
                        _logits = model(_imgs, head_name=head_name)
                    else:
                        _logits = model(_imgs)
                    _preds.extend(_logits.argmax(dim=1).cpu().numpy())
                    _tgt = _sy if model.task == 1 else _ty
                    _targets.extend(_tgt.numpy())
            return {
                "model": model_name,
                "head": head_name or "original",
                "task": "shape" if model.task == 1 else "texture",
                "accuracy": accuracy_score(_targets, _preds),
                "balanced_accuracy": balanced_accuracy_score(_targets, _preds),
                "macro_f1": f1_score(_targets, _preds, average="macro"),
            }

        _loader = selected_dataloader.test_dataloader()
        _rows = [_eval(model_shape, _loader, "shape original")]
        for _h in ["joint", "individual", "both"]:
            _rows.append(_eval(shape_ajive_model, _loader, "shape ajive", _h))
        _rows.append(_eval(model_texture, _loader, "texture original"))
        for _h in ["joint", "individual", "both"]:
            _rows.append(_eval(texture_ajive_model, _loader, "texture ajive", _h))
        results_df = pd.DataFrame(_rows)
        pd.to_pickle(results_df, _clf_cache)

    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for _ax, _metric, _title in zip(
        _axes,
        ["accuracy", "balanced_accuracy", "macro_f1"],
        ["Accuracy", "Balanced Accuracy", "Macro F1"],
    ):
        for _task, _color in [("shape", "steelblue"), ("texture", "darkorange")]:
            _sub = results_df[results_df["task"] == _task].reset_index(drop=True)
            _xlabels = [
                f"{r['model'].split()[-1]}\n{r['head']}" for _, r in _sub.iterrows()
            ]
            _x = [i + (0.2 if _task == "texture" else -0.2) for i in range(len(_sub))]
            _ax.bar(
                _x, _sub[_metric], width=0.35, color=_color, alpha=0.75, label=_task
            )
        _ax.set_xticks(range(max(len(results_df[results_df["task"] == "shape"]), 1)))
        _sub_shape = results_df[results_df["task"] == "shape"].reset_index(drop=True)
        _ax.set_xticklabels(
            [
                f"{r['model'].split()[-1]}\n{r['head']}"
                for _, r in _sub_shape.iterrows()
            ],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        _ax.set_ylim(0, 1.05)
        _ax.set_title(_title, fontweight="bold")
        _ax.grid(axis="y", alpha=0.3)
        if _ax is _axes[0]:
            _ax.legend()

    _fig.suptitle(
        f"Classification Performance — dataset_key={dataset_key}", fontsize=13
    )
    plt.tight_layout()

    mo.vstack([results_df, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prediction Agreement
    """)
    return


@app.cell(hide_code=True)
def _(
    cache_dir,
    dataset_key,
    device,
    mo,
    model_shape,
    model_texture,
    np,
    os,
    pd,
    selected_dataloader,
    shape_ajive_model,
    texture_ajive_model,
    torch,
    tqdm,
):
    _preds_cache = os.path.join(cache_dir, f"all_preds_{dataset_key}.pkl")

    if os.path.exists(_preds_cache):
        all_model_preds = pd.read_pickle(_preds_cache)
    else:
        _keys = [
            "shape_original",
            "shape_ajive_joint",
            "shape_ajive_individual",
            "shape_ajive_both",
            "texture_original",
            "texture_ajive_joint",
            "texture_ajive_individual",
            "texture_ajive_both",
        ]
        _collectors = {k: [] for k in _keys}
        _sl_all, _tl_all = [], []

        with torch.no_grad():
            for _imgs, _sy, _ty in tqdm.tqdm(
                selected_dataloader.test_dataloader(), desc="Collecting predictions"
            ):
                _imgs = _imgs.to(device)
                _collectors["shape_original"].extend(
                    model_shape(_imgs).argmax(1).cpu().numpy()
                )
                _collectors["texture_original"].extend(
                    model_texture(_imgs).argmax(1).cpu().numpy()
                )
                for _h in ["joint", "individual", "both"]:
                    _collectors[f"shape_ajive_{_h}"].extend(
                        shape_ajive_model(_imgs, head_name=_h).argmax(1).cpu().numpy()
                    )
                    _collectors[f"texture_ajive_{_h}"].extend(
                        texture_ajive_model(_imgs, head_name=_h).argmax(1).cpu().numpy()
                    )
                _sl_all.extend(_sy.numpy())
                _tl_all.extend(_ty.numpy())

        _sl = np.array(_sl_all)
        _tl = np.array(_tl_all)
        all_model_preds = {}
        for _k, _p in _collectors.items():
            _p = np.array(_p)
            _true = _sl if "shape" in _k else _tl
            all_model_preds[_k] = pd.DataFrame(
                {
                    "predicted_class": _p,
                    "shape_label": _sl,
                    "texture_label": _tl,
                    "correct": _p == _true,
                }
            )
        pd.to_pickle(all_model_preds, _preds_cache)

    mo.md(
        f"Predictions cached for **{len(all_model_preds)}** model/head combinations "
        f"({len(next(iter(all_model_preds.values())))} test samples each)."
    )
    return (all_model_preds,)


@app.cell(hide_code=True)
def _(mo):
    agreement_task_select = mo.ui.dropdown(
        options=["Shape", "Texture"], value="Shape", label="Task"
    )
    agreement_task_select
    return (agreement_task_select,)


@app.cell(hide_code=True)
def _(
    agreement_task_select,
    all_model_preds,
    np,
    pd,
    plt,
    selected_dataloader,
    sns,
):
    _task = agreement_task_select.value.lower()
    _names = sorted(
        [k for k in all_model_preds if k.startswith(_task)],
        key=lambda x: (0 if "original" in x else 1),
    )
    _short = {
        f"{_task}_original": "Original",
        f"{_task}_ajive_joint": "Joint",
        f"{_task}_ajive_individual": "Indiv.",
        f"{_task}_ajive_both": "Both",
    }
    _class_col = "shape_label" if _task == "shape" else "texture_label"
    _class_names = (
        selected_dataloader.test_dataset.allowed_shapes
        if _task == "shape"
        else selected_dataloader.test_dataset.allowed_textures
    )

    _preds_dict = {n: all_model_preds[n]["predicted_class"].to_numpy() for n in _names}
    _agree_mat = np.array(
        [[(_preds_dict[a] == _preds_dict[b]).mean() for b in _names] for a in _names]
    )
    _lbls = [_short.get(n, n) for n in _names]
    _agree_df = pd.DataFrame(_agree_mat, index=_lbls, columns=_lbls)

    _corr_mat = np.column_stack(
        [all_model_preds[n]["correct"].to_numpy().astype(int) for n in _names]
    )
    _order = np.argsort(_corr_mat.sum(axis=1))
    _corr_sorted = _corr_mat[_order]
    _n_hard = int((_corr_mat.sum(axis=1) == 0).sum())

    _classes = sorted(all_model_preds[_names[0]][_class_col].unique())
    _class_acc_rows = {}
    for _n in _names:
        _df = all_model_preds[_n]
        _class_acc_rows[_short.get(_n, _n)] = [
            _df[_df[_class_col] == c]["correct"].mean() for c in _classes
        ]
    _class_acc_df = pd.DataFrame(
        _class_acc_rows, index=[_class_names[c] for c in _classes]
    ).T

    _fig_agree, (_ax0, _ax1, _ax2) = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(
        _agree_df,
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
        f"{agreement_task_select.value} — Pairwise Prediction Agreement",
        fontweight="bold",
    )

    _im = _ax1.imshow(
        _corr_sorted,
        aspect="auto",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    _ax1.set_xticks(range(len(_lbls)))
    _ax1.set_xticklabels(_lbls)
    _ax1.set_yticks([])
    _ax1.set_ylabel("Sample (easiest → hardest)")
    _ax1.set_title(
        f"{agreement_task_select.value} — Sample Correctness", fontweight="bold"
    )
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
        _class_acc_df,
        annot=True,
        fmt=".2f",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=_ax2,
        cbar_kws={"label": "Accuracy"},
    )
    _ax2.set_xlabel(f"{agreement_task_select.value} class")
    _ax2.set_ylabel("Model")
    _ax2.set_title(
        f"{agreement_task_select.value} — Per-Class Accuracy", fontweight="bold"
    )

    _fig_agree.tight_layout()
    _fig_agree
    return


@app.cell(hide_code=True)
def _(all_model_preds, mo):
    _pred_keys = list(all_model_preds.keys())
    cm_model_a = mo.ui.dropdown(
        options=_pred_keys,
        value=_pred_keys[0],
        label="Model A",
    )
    cm_model_b = mo.ui.dropdown(
        options=_pred_keys,
        value=_pred_keys[4] if len(_pred_keys) > 4 else _pred_keys[-1],
        label="Model B",
    )
    mo.hstack([cm_model_a, cm_model_b])
    return cm_model_a, cm_model_b


@app.cell(hide_code=True)
def _(
    all_model_preds,
    cm_model_a,
    cm_model_b,
    np,
    plt,
    selected_dataloader,
    sns,
):
    def _confusion_matrix(key):
        _df = all_model_preds[key]
        _task = "shape" if key.startswith("shape") else "texture"
        _label_col = "shape_label" if _task == "shape" else "texture_label"
        _class_names = (
            selected_dataloader.test_dataset.allowed_shapes
            if _task == "shape"
            else selected_dataloader.test_dataset.allowed_textures
        )
        _n = len(_class_names)
        _true = _df[_label_col].to_numpy()
        _pred = _df["predicted_class"].to_numpy()
        _mat = np.zeros((_n, _n), dtype=int)
        for _t, _p in zip(_true, _pred):
            _mat[_t, _p] += 1
        return _mat, _class_names

    _fig_cm, (_cax0, _cax1) = plt.subplots(1, 2, figsize=(12, 5))

    for _ax, _key in [(_cax0, cm_model_a.value), (_cax1, cm_model_b.value)]:
        _mat, _cnames = _confusion_matrix(_key)
        _mat_norm = _mat.astype(float) / _mat.sum(axis=1, keepdims=True).clip(min=1)
        sns.heatmap(
            _mat_norm,
            annot=_mat,
            fmt="d",
            vmin=0,
            vmax=1,
            cmap="Blues",
            linewidths=0.5,
            xticklabels=_cnames,
            yticklabels=_cnames,
            ax=_ax,
            cbar_kws={"label": "Row-normalised rate"},
        )
        _acc = _mat.diagonal().sum() / _mat.sum()
        _ax.set_xlabel("Predicted")
        _ax.set_ylabel("True")
        _ax.set_title(f"{_key}\nAccuracy: {_acc:.3f}", fontweight="bold")
        _ax.tick_params(axis="x", rotation=30)
        _ax.tick_params(axis="y", rotation=0)

    _fig_cm.suptitle("Confusion Matrices", fontsize=13)
    _fig_cm.tight_layout()
    _fig_cm
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grad-CAM Preview

    Compare activation maps across up to four model/head combinations.
    Use the correctness filters to restrict the slider to samples where specific models predict correctly.
    """)
    return


@app.cell(hide_code=True)
def _(mo, model_shape, model_texture, shape_ajive_model, texture_ajive_model):
    model_options = {
        "shape original": (model_shape, None),
        "texture original": (model_texture, None),
    }
    model_key_map = {
        "shape original": "shape_original",
        "texture original": "texture_original",
    }
    for _h in ["joint", "individual", "both"]:
        model_options[f"shape ajive {_h}"] = (shape_ajive_model, _h)
        model_options[f"texture ajive {_h}"] = (texture_ajive_model, _h)
        model_key_map[f"shape ajive {_h}"] = f"shape_ajive_{_h}"
        model_key_map[f"texture ajive {_h}"] = f"texture_ajive_{_h}"

    model_keys = list(model_options.keys())

    def _mk(idx):
        return model_keys[idx] if idx < len(model_keys) else model_keys[-1]

    preview_model_1 = mo.ui.dropdown(options=model_keys, value=_mk(0), label="Model 1")
    preview_model_2 = mo.ui.dropdown(options=model_keys, value=_mk(1), label="Model 2")
    preview_model_3 = mo.ui.dropdown(options=model_keys, value=_mk(2), label="Model 3")
    preview_model_4 = mo.ui.dropdown(options=model_keys, value=_mk(3), label="Model 4")

    use_true_class_preview = mo.ui.checkbox(
        label="Use true class for Grad-CAM", value=False
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
    all_model_preds,
    limit_correct_1,
    limit_correct_2,
    limit_correct_3,
    limit_correct_4,
    mo,
    model_key_map,
    np,
    preview_model_1,
    preview_model_2,
    preview_model_3,
    preview_model_4,
    selected_dataloader,
):
    _n_test = len(selected_dataloader.test_dataset)
    _filter_pairs = [
        (limit_correct_1.value, preview_model_1.value),
        (limit_correct_2.value, preview_model_2.value),
        (limit_correct_3.value, preview_model_3.value),
        (limit_correct_4.value, preview_model_4.value),
    ]
    _mask = np.ones(_n_test, dtype=bool)
    _any_filter = False
    for _checked, _mlabel in _filter_pairs:
        if _checked:
            _mkey = model_key_map.get(_mlabel)
            if _mkey in all_model_preds:
                _mask &= all_model_preds[_mkey]["correct"].to_numpy().astype(bool)
                _any_filter = True

    preview_indices = np.flatnonzero(_mask) if _any_filter else np.arange(_n_test)

    preview_pos = mo.ui.slider(
        start=0,
        stop=max(len(preview_indices) - 1, 0),
        step=1,
        value=0,
        label="Sample index",
        full_width=True,
    )
    mo.md(f"Available samples: **{len(preview_indices)}**")
    return preview_indices, preview_pos


@app.cell(hide_code=True)
def _(
    mo,
    model_options,
    plt,
    preview_indices,
    preview_model_1,
    preview_model_2,
    preview_model_3,
    preview_model_4,
    preview_pos,
    preview_ui,
    selected_dataloader,
    torch,
    use_true_class_preview,
):
    _preview_models = [
        preview_model_1.value,
        preview_model_2.value,
        preview_model_3.value,
        preview_model_4.value,
    ]

    if len(preview_indices) == 0:
        _gcam_out = mo.md("No samples match the current filter.")
    else:
        _test_ds = selected_dataloader.test_dataset
        _sample_idx = int(preview_indices[int(preview_pos.value)])
        _img_tensor, _shape_label, _texture_label = _test_ds[_sample_idx]
        _H, _W = _img_tensor.shape[1], _img_tensor.shape[2]
        _img_np = _img_tensor.permute(1, 2, 0).numpy()

        _fig, _axes = plt.subplots(2, 3, figsize=(18, 12))
        _axes[1, 0].axis("off")

        _shape_name = _test_ds.allowed_shapes[int(_shape_label)]
        _texture_name = _test_ds.allowed_textures[int(_texture_label)]
        _axes[0, 0].imshow(_img_np)
        _axes[0, 0].set_title(
            f"Original — sample {_sample_idx}\nShape: {_shape_name} | Texture: {_texture_name}"
        )
        _axes[0, 0].axis("off")

        _positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
        for _pos, _mlabel in zip(_positions, _preview_models):
            _ax = _axes[_pos]
            _model, _head = model_options[_mlabel]
            _dev = next(_model.parameters()).device
            _true_task = int(_shape_label) if _model.task == 1 else int(_texture_label)
            _tc = (
                torch.tensor([_true_task], device=_dev)
                if use_true_class_preview.value
                else None
            )
            _xt = _img_tensor.unsqueeze(0).to(_dev)
            if _head is None:
                _cam, _pred = _model.generate_grad_cam(_xt, target_class=_tc)
            else:
                _cam, _pred = _model.generate_grad_cam(
                    _xt, target_class=_tc, head_name=_head
                )
            _cam_np = (
                torch.nn.functional.interpolate(
                    _cam.detach().cpu().unsqueeze(1),
                    size=(_H, _W),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(1)[0]
                .numpy()
            )
            _class_list = (
                _test_ds.allowed_shapes
                if _model.task == 1
                else _test_ds.allowed_textures
            )
            _pred_idx = int(_pred.item())
            _pred_name = (
                _class_list[_pred_idx]
                if _pred_idx < len(_class_list)
                else str(_pred_idx)
            )
            _ax.imshow(_img_np)
            _im = _ax.imshow(_cam_np, cmap="jet", alpha=0.5)
            _ax.set_title(f"{_mlabel}\nPred: {_pred_name}")
            _ax.axis("off")
            plt.colorbar(_im, ax=_ax, fraction=0.046)

        _fig.suptitle("Grad-CAM Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()
        _gcam_out = mo.vstack([preview_ui, preview_pos, _fig])

    _gcam_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Grad-CAM Similarity

    Pixel-wise similarity between two models' activation maps (Pearson / SSIM / IoU-top-10%).
    The threshold-vs-coverage plot shows how restricting to high-similarity samples affects classification performance.

    **Variants**
    - `pred_each` — each model uses its own predicted class
    - `true` — both models use the ground-truth class
    - `pred_a` — Model B uses Model A's predicted class
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
    controls_1 = mo.hstack([sim_model_1, sim_model_2, sim_target_ui])
    return controls_1, sim_model_1, sim_model_2, sim_target_ui


@app.cell(hide_code=True)
def _(
    cache_dir,
    dataset_key,
    mo,
    model_options,
    np,
    os,
    pd,
    selected_dataloader,
    sim_model_1,
    sim_model_2,
    torch,
    tqdm,
):
    def _pearson(c1, c2):
        _f1, _f2 = c1.ravel(), c2.ravel()
        return float(np.corrcoef(_f1, _f2)[0, 1])

    def _iou_topk(c1, c2, k=0.10):
        _t1, _t2 = np.quantile(c1, 1 - k), np.quantile(c2, 1 - k)
        _m1, _m2 = c1 >= _t1, c2 >= _t2
        _union = (_m1 | _m2).sum()
        return float((_m1 & _m2).sum() / _union) if _union > 0 else 0.0

    def _ssim(c1, c2, C1=1e-4, C2=9e-4):
        _mu1, _mu2 = c1.mean(), c2.mean()
        _s12 = ((c1 - _mu1) * (c2 - _mu2)).mean()
        _denom = (_mu1**2 + _mu2**2 + C1) * (c1.std() ** 2 + c2.std() ** 2 + C2)
        return (
            float(((2 * _mu1 * _mu2 + C1) * (2 * _s12 + C2)) / _denom)
            if _denom != 0
            else 0.0
        )

    def _gcam_raw(model, img_tensor, head_name, target):
        _dev = next(model.parameters()).device
        _tc = torch.tensor([target], device=_dev) if target is not None else None
        _i = img_tensor.unsqueeze(0).to(_dev)
        if head_name is None or not hasattr(model, "head_names"):
            _cam, _pred = model.generate_grad_cam(_i, target_class=_tc)
        else:
            _cam, _pred = model.generate_grad_cam(
                _i, target_class=_tc, head_name=head_name
            )
        return _cam.detach().cpu(), _pred.item()

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
        cache_dir, f"sim_{_m1_key}_vs_{_m2_key}_{dataset_key}.pkl"
    )

    if os.path.exists(_sim_cache_path):
        sim_df = pd.read_pickle(_sim_cache_path)
    else:
        _m1, _h1 = model_options[sim_model_1.value]
        _m2, _h2 = model_options[sim_model_2.value]
        _dataset = selected_dataloader.test_dataset
        _rows = []
        for _didx in tqdm.tqdm(range(len(_dataset)), desc="Grad-CAM similarities"):
            _img_tensor, _sy, _ty = _dataset[_didx]
            _y1, _y2 = int(_sy), int(_ty)
            _true1 = _y1 if getattr(_m1, "task", 1) == 1 else _y2
            _true2 = _y1 if getattr(_m2, "task", 1) == 1 else _y2
            _H, _W = _img_tensor.shape[1], _img_tensor.shape[2]
            try:
                _c1p_t, _pred1 = _gcam_raw(_m1, _img_tensor, _h1, None)
                _c2p_t, _pred2 = _gcam_raw(_m2, _img_tensor, _h2, None)
                _c2p1_t = (
                    _c2p_t
                    if _pred1 == _pred2
                    else _gcam_raw(_m2, _img_tensor, _h2, _pred1)[0]
                )
                _c1t_t = (
                    _c1p_t
                    if _true1 == _pred1
                    else _gcam_raw(_m1, _img_tensor, _h1, _true1)[0]
                )
                _c2t_t = (
                    _c2p_t
                    if _true2 == _pred2
                    else _gcam_raw(_m2, _img_tensor, _h2, _true2)[0]
                )
            except Exception:
                continue
            _r1p = _resize(_c1p_t, _H, _W)
            _r2p = _resize(_c2p_t, _H, _W)
            _r2p1 = _r2p if (_c2p1_t is _c2p_t) else _resize(_c2p1_t, _H, _W)
            _r1t = _r1p if (_c1t_t is _c1p_t) else _resize(_c1t_t, _H, _W)
            _r2t = _r2p if (_c2t_t is _c2p_t) else _resize(_c2t_t, _H, _W)
            _row = {
                "idx": _didx,
                "y1_label": _y1,
                "y2_label": _y2,
                "pred_1": _pred1,
                "pred_2": _pred2,
            }
            _row.update(_sim_metrics(_r1p, _r2p, "pred_each"))
            _row.update(_sim_metrics(_r1t, _r2t, "true"))
            _row.update(_sim_metrics(_r1p, _r2p1, "pred_a"))
            _rows.append(_row)
        sim_df = pd.DataFrame(_rows)
        pd.to_pickle(sim_df, _sim_cache_path)

    mo.md(
        f"Similarity computed for **{len(sim_df)}** images — "
        f"variants: `pred_each`, `true`, `pred_a`."
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
        options=["Shape", "Texture"],
        value="Shape",
        label="Classification task",
    )
    sim_pred_ui = mo.ui.dropdown(
        options=["Model A", "Model B"],
        value="Model A",
        label="Predictions from",
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

    _sim_col = f"{sim_metric_ui.value}_{_target_suffix}"
    _label_col = "y1_label" if sim_task_ui.value == "Shape" else "y2_label"
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

    _vals = sim_df[_sim_col].values
    _thresholds = np.linspace(_vals.min(), _vals.max(), 150)
    _scores, _coverages = [], []
    for _t in _thresholds:
        _sub = sim_df[sim_df[_sim_col] >= _t]
        if len(_sub) < 2:
            continue
        _scores.append(_score_fn(_sub[_label_col], _sub[_pred_col]))
        _coverages.append((sim_df[_sim_col] >= _t).mean())
    _thresholds = _thresholds[: len(_scores)]

    _fig, _ax1 = plt.subplots(figsize=(10, 5))
    _ax2 = _ax1.twinx()
    _ax1.plot(
        _thresholds, _scores, color="steelblue", linewidth=2, label=clf_metric_ui.value
    )
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
    _ax1.set_ylabel(clf_metric_ui.value, color="steelblue", fontsize=11)
    _ax2.set_ylabel("Coverage (fraction accepted)", color="darkorange", fontsize=11)
    _ax1.tick_params(axis="y", labelcolor="steelblue")
    _ax2.tick_params(axis="y", labelcolor="darkorange")
    _lines1, _labels1 = _ax1.get_legend_handles_labels()
    _lines2, _labels2 = _ax2.get_legend_handles_labels()
    _ax1.legend(_lines1 + _lines2, _labels1 + _labels2, loc="center left")
    _ax1.set_title(
        f"{clf_metric_ui.value} vs {_sim_col} threshold\n"
        f"Task: {sim_task_ui.value} | Preds: {sim_pred_ui.value} | "
        f"{sim_model_1.value} vs {sim_model_2.value}",
        fontsize=11,
    )
    _ax1.grid(alpha=0.3)
    plt.tight_layout()

    mo.vstack([controls_1, controls_2, _fig])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## All-Class Grad-CAM Similarity

    Unlike the section above, this analysis does **not** pick a single target class. For a
    fixed pair — **original** vs **AJIVE individual** head, on whichever task is selected below —
    it generates a Grad-CAM map for **every** class on both models and compares the corresponding
    class-maps, yielding **one similarity value per class** for each image (e.g. 4 values for
    shape, 3 for texture).

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
def _(mo):
    allsim_task_ui = mo.ui.dropdown(
        options=["Shape", "Texture"],
        value="Shape",
        label="Task (original vs ajive individual)",
    )
    allsim_task_ui
    return (allsim_task_ui,)


@app.cell(hide_code=True)
def _(
    allsim_task_ui,
    cache_dir,
    dataset_key,
    mo,
    model_options,
    np,
    os,
    pd,
    selected_dataloader,
    torch,
    tqdm,
):
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

    allsim_task_key = "shape" if allsim_task_ui.value == "Shape" else "texture"
    _mA, _hA = model_options[f"{allsim_task_key} original"]
    _mB, _hB = model_options[f"{allsim_task_key} ajive individual"]
    _dataset = selected_dataloader.test_dataset

    _probe_img = _dataset[0][0]
    with torch.no_grad():
        _dev_a = next(_mA.parameters()).device
        num_classes = int(_mA(_probe_img.unsqueeze(0).to(_dev_a)).shape[1])

    _allsim_cache_path = os.path.join(
        cache_dir,
        f"sim_allclass_{allsim_task_key}_orig_vs_ajive_indiv_{dataset_key}.pkl",
    )

    if os.path.exists(_allsim_cache_path):
        allsim_df = pd.read_pickle(_allsim_cache_path)
    else:
        _rows = []
        for _didx in tqdm.tqdm(
            range(len(_dataset)), desc="All-class GradCAM similarities"
        ):
            _image_tensor, _shape_label, _texture_label = _dataset[_didx]
            _y1 = int(_shape_label)
            _y2 = int(_texture_label)
            _H, _W = _image_tensor.shape[1], _image_tensor.shape[2]
            try:
                _pred1 = None
                _pred2 = None
                _row = {"idx": _didx, "shape_label": _y1, "texture_label": _y2}
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
        f"across **{num_classes}** classes "
        f"({allsim_task_key} original vs AJIVE individual)."
    )
    return allsim_df, allsim_task_key, num_classes


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
        label="Aggregate class-similarities by",
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
    allsim_task_key,
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
    _label_col = f"{allsim_task_key}_label"

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
        "predicted class (Model A)": lambda a: a[
            _rows_idx, allsim_df["pred_1"].to_numpy().clip(0, num_classes - 1)
        ],
        "true class": lambda a: a[
            _rows_idx, allsim_df[_label_col].to_numpy().clip(0, num_classes - 1)
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
        f"{allsim_task_key} original"
        if _pred_col == "pred_1"
        else f"{allsim_task_key} ajive individual"
    )
    _title_line1 = (
        f"{_clf_label} vs {_agg}({_metric}) threshold — "
        f"{allsim_task_key.capitalize()} | Preds: {_pred_label}"
    )
    _title_line2 = (
        f"all-class similarity | {allsim_task_key} original vs "
        f"{allsim_task_key} ajive individual"
    )
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
