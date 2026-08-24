import itertools
from typing import Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from elements.classes import ElementDataset


class Single_Element_dataset(Dataset):
    """Wrapper around ElementDataset for single-object images.

    Returns one image and two labels:
    - shape label in [0, 3]
    - texture label in [0, 2]
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
        element_n: int = 1,
        allowed_shapes: Optional[List[str]] = None,
        allowed_colors: Optional[List[str]] = None,
        allowed_textures: Optional[List[str]] = None,
        element_seed: int = 42,
        loc_seed: int = 123,
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

        # 12 classes: all shape x texture combinations with wildcard color.
        self.class_configs = [
            {"shape": shape, "color": None, "texture": texture}
            for shape, texture in itertools.product(
                self.allowed_shapes, self.allowed_textures
            )
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

        allowed = {
            "shapes": self.allowed_shapes,
            "colors": self.allowed_colors,
            "textures": self.allowed_textures,
        }

        self.dataset = ElementDataset(
            allowed=allowed,
            class_configs=self.class_configs,
            n=split_sizes[split],
            img_size=img_size,
            element_n=element_n,
            element_size=element_size,
            element_size_delta=element_size_delta,
            element_seed=element_seed + split_offset,
            loc_seed=loc_seed + split_offset,
        )

        # Precompute labels from element sampling seeds. This avoids generating images
        # when computing class balance for samplers.
        self.shape_labels, self.texture_labels, self.combo_labels = (
            self._precompute_labels()
        )

    def _precompute_labels(self):
        shape_labels = np.zeros(len(self.dataset), dtype=np.int64)
        texture_labels = np.zeros(len(self.dataset), dtype=np.int64)
        combo_labels = np.zeros(len(self.dataset), dtype=np.int64)

        for idx in range(len(self.dataset)):
            element_config = ElementDataset.choose_element_configs(
                n=1,
                allowed_sizes=self.dataset.allowed["sizes"],
                allowed_shapes=self.dataset.allowed["shapes"],
                allowed_colors=self.dataset.allowed["colors"],
                allowed_textures=self.dataset.allowed["textures"],
                seed=int(self.dataset.element_seeds[idx]),
                allowed_combinations=self.dataset.allowed_combinations,
            )[0]

            shape = element_config["shape"]
            texture = element_config["texture"]

            shape_idx = self.shape_to_idx[shape]
            texture_idx = self.texture_to_idx[texture]

            shape_labels[idx] = shape_idx
            texture_labels[idx] = texture_idx
            combo_labels[idx] = shape_idx * len(self.allowed_textures) + texture_idx

        return shape_labels, texture_labels, combo_labels

    def __len__(self):
        return len(self.dataset)

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
        element_n: int = 1,
        allowed_shapes: Optional[List[str]] = None,
        allowed_colors: Optional[List[str]] = None,
        allowed_textures: Optional[List[str]] = None,
        element_seed: int = 42,
        loc_seed: int = 123,
        train_transform=None,
        transform=None,
        use_train_sampler: bool = True,
        sampler_target: str = "shape",
    ):
        super().__init__()

        assert sampler_target in ["shape", "texture", "combo"], (
            'sampler_target must be one of "shape", "texture", "combo"'
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
        if self.sampler_target == "texture":
            return dataset.texture_labels
        return dataset.combo_labels

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
