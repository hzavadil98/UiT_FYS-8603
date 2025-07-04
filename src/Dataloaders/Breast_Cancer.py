import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pydicom import dcmread
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class Breast_Cancer_Dataset(Dataset):
    """
    Dotaset to load breast wise images - list withl CC and MLO images and one label for both.
    in this setting can only load the whole dataset, only separated to splits, if in future want to load only one laterality must redo
    """

    def __init__(
        self,
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        split=None,
        transform=None,
        # laterality: str = None,
    ):
        """
        root_folder
            -> modified_breast-level_annotations.csv
            -> New_512
                        -> image_id.dicom

        root_folder: path to folder containing the "images" folder and csv-file
        You can select the split to extract specific data from the dataset, outputs  4 images for the patient(LCC,LMLO,RCC,RMLO) and lef and right labels
        """
        super().__init__()

        assert split in ["training", "test", "validation", None], (
            'split must be either "training" or "test" or "validation" or None'
        )
        # assert laterality in ["L", "R", None], 'view must be either "L" or "R"'

        self.split = split
        self.imagefolder_path = imagefolder_path
        self.root_folder = root_folder
        self.transforms = transform
        # self.laterality = laterality

        annotation_csv = pd.read_csv(os.path.join(root_folder, annotation_csv))

        if split is not None:
            splitBool = True
            splitBool = annotation_csv["split"] == split
            # selects only the rows corresponding to the split
            self.annotation = annotation_csv.loc[splitBool]
        # if laterality is not None:
        #    lateralityBool = True
        #    lateralityBool = self.annotation["laterality"] == self.laterality
        #    self.annotation = self.annotation.loc[lateralityBool]

        # finds all the unique study_ids = "patient_ids"
        self.patient_ids = self.annotation["study_id"].unique()
        # keep the number of distinct patient ids
        self.n_patients = len(self.patient_ids)
        # Stack patient_ids on itself
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
        # finds two labels for each patient_id - L,R
        self.labels = (
            self.annotation.groupby(["study_id", "laterality"])["breast_birads"]
            .unique()
            .unstack()
        )
        # find two densities for each patient_id - L,R
        self.densities = (
            self.annotation.groupby(["study_id", "laterality"])["breast_density"]
            .unique()
            .unstack()
        )

        # sorts the labels according to the patient_ids
        self.labels = self.labels.reindex(self.patient_ids)
        self.densities = self.densities.reindex(self.patient_ids)

    def __len__(self):
        return 2 * len(self.patient_ids)

    def __getitem__(self, idx):
        laterality = "R" if idx >= self.n_patients else "L"
        idx = idx % self.n_patients
        patient_id = self.patient_ids[idx]
        data_lines = self.annotation[
            (self.annotation["study_id"] == patient_id)
            & (self.annotation["laterality"] == laterality)
        ]

        label = torch.tensor(self.labels.loc[patient_id, laterality][0])
        density = torch.tensor(self.densities.loc[patient_id, laterality][0])

        images = []

        for view in ["CC", "MLO"]:
            image_id = data_lines[
                (data_lines["laterality"] == laterality)
                & (data_lines["view_position"] == view)
            ]["image_id"].values[0]
            image_path = os.path.join(
                self.root_folder + self.imagefolder_path, image_id + ".dicom"
            )
            try:
                image = dcmread(image_path).pixel_array.astype(np.float32)
                image = torch.from_numpy(image).unsqueeze(0)  # Add channel dimension
                if self.transforms is not None:
                    image = self.transforms(image)
                images.append(image)
            except Exception as e:
                print(f"Error reading image {image_path}: {e}")
                continue

        return images, label, density

    def plot(self, idx):
        images, label, density = self.__getitem__(idx)
        # Transform images to the interval (0, 1)
        images = [
            (image - image.min()) / (image.max() - image.min()) for image in images
        ]
        fig, axs = plt.subplots(1, 2)

        for i in range(2):
            # axs[i].imshow(images[i].permute(1, 2, 0))
            axs[i].imshow(images[i][0])
            axs[i].set_title(f"Label: {label.item()}, Density: {density.item()}")
        plt.show()


class Breast_Cancer_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        batch_size: int,
        num_workers: int,
        train_transform=None,
        transform=None,
        task: int = 1,  # 1 for birads classification, 2 for density classification
    ):
        super().__init__()

        self.root_folder = root_folder
        self.annotation_csv = annotation_csv
        self.imagefolder_path = imagefolder_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.transform = transform
        self.task = task
        assert task in [1, 2], (
            "task must be either 1 (birads classification) or 2 (density classification)"
        )

        self.train_dataset = Breast_Cancer_Dataset(
            self.root_folder,
            self.annotation_csv,
            self.imagefolder_path,
            split="training",
            transform=self.train_transform,
        )
        self.val_dataset = Breast_Cancer_Dataset(
            self.root_folder,
            self.annotation_csv,
            self.imagefolder_path,
            split="validation",
            transform=self.transform,
        )
        self.test_dataset = Breast_Cancer_Dataset(
            self.root_folder,
            self.annotation_csv,
            self.imagefolder_path,
            split="test",
            transform=self.transform,
        )
        if self.task == 1:
            labels = self.train_dataset.labels
        else:
            labels = self.train_dataset.densities

        targets = np.concatenate((labels["L"].values, labels["R"].values))
        targets = np.array([x[0] for x in targets])
        class_sample_count = np.array(
            [len(np.where(targets == t)[0]) for t in np.unique(targets)]
        )
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in targets])
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
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def plot(self, idx, dataset="train"):
        if dataset == "train":
            print(f"Length of train dataset: {len(self.train_dataset)}")
            self.train_dataset.plot(idx)
        elif dataset == "val":
            print(f"Length of val dataset: {len(self.val_dataset)}")
            self.val_dataset.plot(idx)
        elif dataset == "test":
            print(f"Length of test dataset: {len(self.test_dataset)}")
            self.test_dataset.plot(idx)
        else:
            raise ValueError("dataset must be either 'train', 'val' or 'test'")
