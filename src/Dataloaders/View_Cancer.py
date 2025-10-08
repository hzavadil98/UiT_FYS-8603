from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pydicom import dcmread
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class View_Cancer_dataset(Dataset):
    def __init__(
        self,
        root_folder,
        annotation_csv,
        imagefolder_path,
        image_format: str,
        norm_kind: str = "dataset_zscore",
        split=None,
        transform=None,
        view: str = None,
        laterality: str = None,
    ):
        """
        root_folder
            -> modified_breast-level_annotations.csv
            -> New_512
                        -> image_id.dicom
            -> images_png
                        -> image_id.png

        root_folder: path to folder containing the "images" folder and csv-file
        You can select the split to extract specific data from the dataset, outputs 1 image specified by the view parameter and its label
        """
        super().__init__()

        assert split in ["training", "test", "validation", None], (
            'split must be either "training" or "test" or "validation" or None'
        )
        assert view in ["CC", "MLO", None], 'laterality must be either "CC" or "MLO"'
        assert laterality in ["L", "R", None], 'view must be either "L" or "R"'
        assert image_format in ["png", "dicom"], (
            'image_format must be either "png" or "dicom"'
        )
        assert norm_kind in ["dataset_zscore", "zscore", "minmax", None], (
            'norm_kind must be either "dataset_zscore" or "zscore" or "minmax"'
        )

        self.split = split
        self.imagefolder_path = Path(imagefolder_path)
        self.root_folder = Path(root_folder)
        self.image_format = image_format
        self.norm_kind = norm_kind
        self.transforms = transform
        self.view = view
        self.laterality = laterality

        annotation_csv = pd.read_csv(self.root_folder / annotation_csv)

        if split is not None:
            splitBool = True
            splitBool = annotation_csv["split"] == split
            # selects only the rows corresponding to the split
            self.annotation = annotation_csv.loc[splitBool]
        # selects only the rows corresponding to the view and laterality
        if view is not None:
            viewBool = True
            viewBool = self.annotation["view_position"] == self.view
            self.annotation = self.annotation.loc[viewBool]
        if laterality is not None:
            lateralityBool = True
            lateralityBool = self.annotation["laterality"] == self.laterality
            self.annotation = self.annotation.loc[lateralityBool]
        # finds all the unique study_ids = "patient_ids"
        self.image_ids = self.annotation["image_id"].values

        self.label_map = {
            "BI-RADS 1": 0,
            "BI-RADS 2": 1,
            "BI-RADS 3": 2,
            "BI-RADS 4": 3,
            "BI-RADS 5": 4,
        }
        self.density_map = {
            "DENSITY A": 0,
            "DENSITY B": 1,
            "DENSITY C": 2,
            "DENSITY D": 3,
        }
        # maps the labels to integers
        self.annotation.loc[:, "breast_birads"] = self.annotation["breast_birads"].map(
            self.label_map
        )
        self.annotation.loc[:, "breast_density"] = self.annotation[
            "breast_density"
        ].map(self.density_map)
        # gets image labels
        self.labels = self.annotation["breast_birads"].values
        self.densities = self.annotation["breast_density"].values

    def load_img_to_tensor(self, image_id) -> torch.Tensor:
        """Load a grayscale image from disk and convert it to a tensor of shape (3, H, W).

        Args:
            image_id (str): The ID of the image to load.

        Returns:
            torch.Tensor: The loaded image as a tensor.
        """
        if self.image_format == "dicom":
            image_path = (
                self.root_folder / self.imagefolder_path / (image_id + ".dicom")
            )
            try:
                image = torch.from_numpy(
                    dcmread(image_path).pixel_array.astype(np.float32)
                )
                image = image.unsqueeze(0).repeat(3, 1, 1)
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")

        else:
            image_path = self.root_folder / self.imagefolder_path / (image_id + ".png")
            try:
                image = torch.from_numpy(iio.imread(image_path).astype(np.float32))
                image = image.unsqueeze(0).repeat(3, 1, 1)
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")

        return image

    def normalise_image(self, image: torch.Tensor, norm_kind: str) -> torch.Tensor:
        if norm_kind == "minmax":
            if self.image_format == "dicom":
                image = image / 65535.0
            else:
                image = image / 255.0

        elif norm_kind == "zscore":
            image = (image - image.mean()) / image.std()

        elif norm_kind == "dataset_zscore":
            if self.image_format == "dicom":
                image = (
                    (image - 923.4552) / 2035.7942
                )  # approx mean and std calculated from the train dataset of New_512 imgs using the weighted sampler, (unweighted - single pass - 781.06, 1184.94) from suaiba got 781.0543 and 1537.8235 for some reason
            else:
                image = (
                    (image - 108.3328) / 69.6431
                )  # mean and std calculated from the train dataset of images_png with weighted sampler (unweighted - single pass - 104.81, 66.57)

        else:
            pass

        return image

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        label = self.labels[idx]

        density = self.densities[idx]

        image = self.load_img_to_tensor(image_id)

        image = self.normalise_image(image, norm_kind=self.norm_kind)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, label, density

    def plot(self, idx):
        # plots the image and a histogram of the pixel values into one figure
        image, label = self.__getitem__(idx)
        plotimage = (image - image.min()) / (image.max() - image.min())
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(plotimage.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Label: {label}, view: {self.view}, laterality: {self.laterality}")
        plt.subplot(1, 2, 2)
        plt.hist(image.flatten(), bins=50)
        plt.yscale("log")
        plt.title("Pixel value distribution")
        plt.show()


class View_Cancer_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        image_format: str,
        norm_kind: str = "dataset_zscore",
        batch_size: int = 32,
        num_workers: int = 4,
        view: str = None,
        laterality: str = None,
        train_transform=None,
        transform=None,
        task: int = 1,
    ):
        super().__init__()
        self.view = view
        self.laterality = laterality
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.transform = transform
        assert task in [1, 2], "Task must be 1 (cancer) or 2 (density)"
        self.task = task

        self.train_dataset = View_Cancer_dataset(
            root_folder=root_folder,
            annotation_csv=annotation_csv,
            imagefolder_path=imagefolder_path,
            image_format=image_format,
            norm_kind=norm_kind,
            split="training",
            transform=self.train_transform,
            view=view,
            laterality=laterality,
        )
        self.val_dataset = View_Cancer_dataset(
            root_folder=root_folder,
            annotation_csv=annotation_csv,
            imagefolder_path=imagefolder_path,
            image_format=image_format,
            norm_kind=norm_kind,
            split="validation",
            transform=self.transform,
            view=view,
            laterality=laterality,
        )
        self.test_dataset = View_Cancer_dataset(
            root_folder=root_folder,
            annotation_csv=annotation_csv,
            imagefolder_path=imagefolder_path,
            image_format=image_format,
            norm_kind=norm_kind,
            split="test",
            transform=self.transform,
            view=view,
            laterality=laterality,
        )

        if self.task == 1:
            labels = self.train_dataset.labels
        else:
            labels = self.train_dataset.densities

        class_sample_count = np.array(
            [len(np.where(labels == t)[0]) for t in np.unique(labels)]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        self.train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
        )

    def plot(self, idx):
        self.train_dataset.plot(idx)
        self.val_dataset.plot(idx)
