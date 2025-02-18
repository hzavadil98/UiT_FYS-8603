import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pydicom import dcmread
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class Patient_Cancer_Dataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        split=None,
        transform=None,
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

        self.split = split
        self.imagefolder_path = imagefolder_path
        self.root_folder = root_folder
        self.transforms = transform

        annotation_csv = pd.read_csv(os.path.join(root_folder, annotation_csv))

        splitBool = True
        if split is not None:
            splitBool = annotation_csv["split"] == split
        # selects only the rows corresponding to the split
        self.annotation = annotation_csv.loc[splitBool]
        # finds all the unique study_ids = "patient_ids"
        self.patient_ids = self.annotation["study_id"].unique()

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
        self.labels = (
            self.annotation.groupby(["study_id", "laterality"])["breast_birads"]
            .unique()
            .unstack()
        )
        # sorts the labels according to the patient_ids
        self.labels = self.labels.reindex(self.patient_ids)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        data_lines = self.annotation[self.annotation["study_id"] == patient_id]

        # returns left and right labels
        labels = torch.tensor(
            [self.labels.loc[patient_id, "L"][0], self.labels.loc[patient_id, "R"][0]]
        )

        images = []
        for laterality in ["L", "R"]:
            for view in ["CC", "MLO"]:
                image_id = data_lines[
                    (data_lines["laterality"] == laterality)
                    & (data_lines["view_position"] == view)
                ]["image_id"].values[0]
                # image_id = data_lines.iloc[i]["image_id"]
                image_path = os.path.join(
                    self.root_folder + self.imagefolder_path, image_id + ".dicom"
                )
                try:
                    image = dcmread(image_path).pixel_array
                    image = np.array(image, dtype=np.float32)
                    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
                    if self.transforms is not None:
                        image = self.transforms(image)
                    images.append(image)
                except Exception as e:
                    print(f"Error reading image {image_path}: {e}")
                    continue

        return images, labels

    def plot(self, idx):
        images, labels = self.__getitem__(idx)
        # Transform images to the interval (0, 1)
        images = [(image - image.min()) / (image.max() - image.min()) for image in images]
        fig, axs = plt.subplots(2, 2)
        
        for i in range(2):
            for j in range(2):
                axs[i, j].imshow(images[i * 2 + j].permute(1, 2, 0))
                axs[i, j].set_title(f"Label: {labels[i].item()}")
        plt.show()


class Patient_Cancer_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        batch_size=16,
        num_workers=8,
    ):
        """
        root_folder
            -> breast-level_annotations.csv
            -> Preprocessed
                        -> image_id.dicom



        root_folder: path to folder containing the "images" folder and csv-file
        imagefolder_path: path to folder containing the images
        annotation_csv: name of the csv in root_folder
        split: Either "training" or "validation" or "test" split of dataset.
        transform:
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        ############## do I want these transforms here? removed flipping the images ##############
        self.train_transform = T.Compose(
            [
                T.ToTensor(),
                T.RandomRotation(degrees=10),
                T.Normalize(
                    mean=[781.0543, 781.0543, 781.0543],
                    std=[1537.8235, 1537.8235, 1537.8235],
                ),
            ]
        )
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[781.0543, 781.0543, 781.0543],
                    std=[1537.8235, 1537.8235, 1537.8235],
                ),
            ]
        )

        self.train_dataset = Patient_Cancer_Dataset(
            root_folder,
            annotation_csv,
            imagefolder_path,
            split="training",
            transform=self.train_transform,
        )
        self.val_dataset = Patient_Cancer_Dataset(
            root_folder,
            annotation_csv,
            imagefolder_path,
            split="validation",
            transform=self.transform,
        )
        self.test_dataset = Patient_Cancer_Dataset(
            root_folder,
            annotation_csv,
            imagefolder_path,
            split="test",
            transform=self.transform,
        )

        # Create a WeightedRandomSampler to handle the class imbalance, taking max birads for a patient and using it as the target
        labels = self.train_dataset.labels
        targets = np.array([x[0] for x in labels.max(axis=1).values])

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