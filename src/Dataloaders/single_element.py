from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from elements.classes import ElementDataset


class Single_Element_dataset(Dataset):
    """Wrapper around ElementDataset for multi-object images.

    Returns one image and two labels:
    - task1: majority shape label
    - task2: majority texture label

    Samples where either task is a tie are rejected and re-sampled.
    """

    def __init__(
        self,
        split: str = "training",
        n_train: int = 5000,
        n_val: int = 1000,
        n_test: int = 1000,
        img_size: int = 224,
        element_size: int = 96,
        element_size_delta: int = 24,
        element_n: int = 5,
        allowed_shapes: Optional[List[str]] = None,
        allowed_colors: Optional[List[str]] = None,
        allowed_textures: Optional[List[str]] = None,
        element_seed: int = 42,
        loc_seed: int = 123,
        max_resample_attempt_factor: int = 50,
        transform=None,
    ):
        super().__init__()

        assert split in ["training", "validation", "test"], (
            'split must be "training", "validation", or "test"'
        )

        self.split = split
        self.transform = transform

        self.allowed_shapes = allowed_shapes or ["square", "circle", "triangle", "plus"]
        self.allowed_colors = allowed_colors or ["red", "green", "blue"]
        self.allowed_textures = allowed_textures or [
            "solid",
            "spots_polka",
            "stripes_diagonal",
        ]
        self.element_n = element_n
        self.max_resample_attempt_factor = max_resample_attempt_factor

        # Keep class configs minimal: labels are computed from majority statistics,
        # not from this class list.
        self.class_configs = [
            {"shape": shape, "color": None, "texture": None}
            for shape in self.allowed_shapes
        ]

        self.shape_to_idx = {shape: i for i, shape in enumerate(self.allowed_shapes)}
        self.texture_to_idx = {
            texture: i for i, texture in enumerate(self.allowed_textures)
        }

        split_sizes = {
            "training": n_train,
            "validation": n_val,
            "test": n_test,
        }

        # Use deterministic split-specific seeds so the 3 splits differ.
        split_offsets = {
            "training": 0,
            "validation": 10_000,
            "test": 20_000,
        }
        split_offset = split_offsets[split]
        split_size = split_sizes[split]

        allowed = {
            "shapes": self.allowed_shapes,
            "colors": self.allowed_colors,
            "textures": self.allowed_textures,
        }

        self.dataset = ElementDataset(
            allowed=allowed,
            class_configs=self.class_configs,
            n=split_size,
            img_size=img_size,
            element_n=element_n,
            element_size=element_size,
            element_size_delta=element_size_delta,
            element_seed=element_seed + split_offset,
            loc_seed=loc_seed + split_offset,
        )

        (
            accepted_element_seeds,
            accepted_loc_seeds,
            self.shape_labels,
            self.texture_labels,
        ) = self._sample_non_tie_seed_bank(
            n_samples=split_size,
            base_element_seed=element_seed + split_offset,
            base_loc_seed=loc_seed + split_offset,
        )

        self.dataset.element_seeds = accepted_element_seeds
        self.dataset.loc_seeds = accepted_loc_seeds
        self.combo_labels = (
            self.shape_labels * len(self.allowed_textures) + self.texture_labels
        )

    @staticmethod
    def _majority_label_or_none(indices: List[int], n_classes: int) -> Optional[int]:
        counts = np.bincount(indices, minlength=n_classes)
        max_count = counts.max()
        winners = np.flatnonzero(counts == max_count)
        if len(winners) != 1:
            return None
        return int(winners[0])

    def _sample_non_tie_seed_bank(
        self,
        n_samples: int,
        base_element_seed: int,
        base_loc_seed: int,
    ):
        element_rng = np.random.default_rng(base_element_seed)
        loc_rng = np.random.default_rng(base_loc_seed)

        accepted_element_seeds = []
        accepted_loc_seeds = []
        shape_labels = []
        texture_labels = []

        max_attempts = n_samples * self.max_resample_attempt_factor
        attempts = 0

        while len(accepted_element_seeds) < n_samples and attempts < max_attempts:
            attempts += 1

            candidate_element_seed = int(element_rng.integers(0, 1_000_000))
            candidate_loc_seed = int(loc_rng.integers(0, 1_000_000))

            element_configs = ElementDataset.choose_element_configs(
                n=self.element_n,
                allowed_sizes=self.dataset.allowed["sizes"],
                allowed_shapes=self.dataset.allowed["shapes"],
                allowed_colors=self.dataset.allowed["colors"],
                allowed_textures=self.dataset.allowed["textures"],
                seed=candidate_element_seed,
                allowed_combinations=self.dataset.allowed_combinations,
            )

            shape_indices = [self.shape_to_idx[cfg["shape"]] for cfg in element_configs]
            texture_indices = [
                self.texture_to_idx[cfg["texture"]] for cfg in element_configs
            ]

            shape_majority = self._majority_label_or_none(
                shape_indices, len(self.allowed_shapes)
            )
            texture_majority = self._majority_label_or_none(
                texture_indices, len(self.allowed_textures)
            )

            if shape_majority is None or texture_majority is None:
                continue

            accepted_element_seeds.append(candidate_element_seed)
            accepted_loc_seeds.append(candidate_loc_seed)
            shape_labels.append(shape_majority)
            texture_labels.append(texture_majority)

        if len(accepted_element_seeds) < n_samples:
            raise RuntimeError(
                "Could not sample enough non-tie images. "
                "Increase max_resample_attempt_factor, reduce element_n, "
                "or reduce the number of classes."
            )

        return (
            np.array(accepted_element_seeds, dtype=np.int64),
            np.array(accepted_loc_seeds, dtype=np.int64),
            np.array(shape_labels, dtype=np.int64),
            np.array(texture_labels, dtype=np.int64),
        )

    def __len__(self):
        return len(self.shape_labels)

    def __getitem__(self, idx):
        image, _class_oh = self.dataset[idx]

        if self.transform is not None:
            image = self.transform(image)

        shape_label = int(self.shape_labels[idx])
        texture_label = int(self.texture_labels[idx])

        return image, shape_label, texture_label


class Single_Element_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        n_train: int = 5000,
        n_val: int = 1000,
        n_test: int = 1000,
        img_size: int = 224,
        element_size: int = 96,
        element_size_delta: int = 24,
        element_n: int = 5,
        allowed_shapes: Optional[List[str]] = None,
        allowed_colors: Optional[List[str]] = None,
        allowed_textures: Optional[List[str]] = None,
        element_seed: int = 42,
        loc_seed: int = 123,
        max_resample_attempt_factor: int = 50,
        train_transform=None,
        transform=None,
        use_train_sampler: bool = True,
        sampler_target: str = "shape",
    ):
        super().__init__()

        assert sampler_target in ["shape", "texture"], (
            'sampler_target must be one of "shape" or "texture"'
        )

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_train_sampler = use_train_sampler
        self.sampler_target = sampler_target

        common_kwargs: Dict = {
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "img_size": img_size,
            "element_size": element_size,
            "element_size_delta": element_size_delta,
            "element_n": element_n,
            "allowed_shapes": allowed_shapes,
            "allowed_colors": allowed_colors,
            "allowed_textures": allowed_textures,
            "element_seed": element_seed,
            "loc_seed": loc_seed,
            "max_resample_attempt_factor": max_resample_attempt_factor,
        }

        self.train_dataset = Single_Element_dataset(
            split="training", transform=train_transform, **common_kwargs
        )
        self.val_dataset = Single_Element_dataset(
            split="validation", transform=transform, **common_kwargs
        )
        self.test_dataset = Single_Element_dataset(
            split="test", transform=transform, **common_kwargs
        )

        train_targets = self._get_train_targets(self.train_dataset)
        class_sample_count = np.array(
            [len(np.where(train_targets == t)[0]) for t in np.unique(train_targets)]
        )
        class_weights = 1.0 / class_sample_count
        sample_weights = np.array([class_weights[t] for t in train_targets])
        sample_weights = torch.from_numpy(sample_weights).double()

        self.train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    def _get_train_targets(self, dataset: Single_Element_dataset) -> np.ndarray:
        if self.sampler_target == "shape":
            return dataset.shape_labels
        return dataset.texture_labels

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler if self.use_train_sampler else None,
            shuffle=False if self.use_train_sampler else True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
