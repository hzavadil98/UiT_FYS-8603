import os

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
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        split=None,
        transform=None,
        view: int = 0,
    ):
        """
        root_folder
            -> modified_breast-level_annotations.csv
            -> New_512
                        -> image_id.dicom

        root_folder: path to folder containing the "images" folder and csv-file
        You can select the split to extract specific data from the dataset, outputs 1 image specified by the view parameter and its label
        """
        super().__init__()

        assert split in ["training", "test", "validation", None], (
            'split must be either "training" or "test" or "validation" or None'
        )

        self.split = split
        self.imagefolder_path = imagefolder_path
        self.root_folder = root_folder
        self.transforms = transform
        self.view = view
        self.laterality = "L" if view < 2 else "R"
        self.view_position = "CC" if view % 2 == 0 else "MLO"
        annotation_csv = pd.read_csv(os.path.join(root_folder, annotation_csv))

        splitBool = True
        if split is not None:
            splitBool = annotation_csv["split"] == split
        # selects only the rows corresponding to the split
        self.annotation = annotation_csv.loc[splitBool]
        # selects only the rows corresponding to the view and laterality
        laterality_bool = self.annotation["laterality"] == self.laterality
        view_position_bool = self.annotation["view_position"] == self.view_position
        self.annotation = self.annotation.loc[laterality_bool & view_position_bool]

        # finds all the unique study_ids = "patient_ids"
        self.image_ids = self.annotation["image_id"].values

        self.label_map = {
            "BI-RADS 1": 0,
            "BI-RADS 2": 1,
            "BI-RADS 3": 2,
            "BI-RADS 4": 3,
            "BI-RADS 5": 4,
        }
        # maps the labels to integers
        self.annotation.loc[:, "breast_birads"] = self.annotation["breast_birads"].map(
            self.label_map
        )
        # finds two labels for each patient_id - L,R
        self.labels = self.annotation["breast_birads"].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        label = self.labels[idx]

        image_path = os.path.join(
            self.root_folder + self.imagefolder_path, image_id + ".dicom"
        )
        try:
            image = dcmread(image_path).pixel_array
            image = np.array(image, dtype=np.float32)
            image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
            if self.transforms is not None:
                image = self.transforms(image)
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")

        return image, label

    def plot(self, idx):
        # plots the image and a histogram of the pixel values into one figure
        image, label = self.__getitem__(idx)
        plotimage = (image - image.min()) / (image.max() - image.min())
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(plotimage.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Label: {label}")
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
        view: int,
        batch_size: int,
        num_workers: int,
        train_transform=None,
        transform=None,
    ):
        super().__init__()
        self.view = view
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.transform = transform

        self.train_dataset = View_Cancer_dataset(
            root_folder=root_folder,
            annotation_csv=annotation_csv,
            imagefolder_path=imagefolder_path,
            split="training",
            transform=self.train_transform,
            view=view,
        )
        self.val_dataset = View_Cancer_dataset(
            root_folder=root_folder,
            annotation_csv=annotation_csv,
            imagefolder_path=imagefolder_path,
            split="validation",
            transform=self.transform,
            view=view,
        )
        self.test_dataset = View_Cancer_dataset(
            root_folder=root_folder,
            annotation_csv=annotation_csv,
            imagefolder_path=imagefolder_path,
            split="test",
            transform=self.transform,
            view=view,
        )

        labels = self.train_dataset.labels
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
