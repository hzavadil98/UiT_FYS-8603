from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pydicom import dcmread
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class Patient_Cancer_Dataset(Dataset):
    def __init__(
        self,
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        image_format: str = "dicom",
        norm_kind: str = "dataset_zscore",
        split=None,
        transform=None,
        cancer_label_type: str = "birads",
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
        assert image_format in ["png", "dicom"], (
            'image_format must be either "png" or "dicom"'
        )
        assert norm_kind in ["dataset_zscore", "zscore", "minmax", None], (
            'norm_kind must be either "dataset_zscore" or "zscore" or "minmax"'
        )
        assert cancer_label_type in ["birads", "diagnosis", "binary"], (
            'cancer_label_type must be either "birads", "diagnosis" or "binary"'
        )

        self.split = split
        self.imagefolder_path = Path(imagefolder_path)
        self.root_folder = Path(root_folder)
        self.image_format = image_format
        self.norm_kind = norm_kind
        self.transforms = transform
        annotation_csv = pd.read_csv(self.root_folder / annotation_csv)

        splitBool = True
        if split is not None:
            splitBool = annotation_csv["split"] == split
        # selects only the rows corresponding to the split
        self.annotation = annotation_csv.loc[splitBool]
        # finds all the unique study_ids = "patient_ids"
        self.patient_ids = self.annotation["study_id"].unique()

        if cancer_label_type == "birads":
            self.label_map = {
                "BI-RADS 1": 0,
                "BI-RADS 2": 1,
                "BI-RADS 3": 2,
                "BI-RADS 4": 3,
                "BI-RADS 5": 4,
            }
        elif cancer_label_type == "diagnosis":
            self.label_map = {
                "BI-RADS 1": 0,
                "BI-RADS 2": 1,
                "BI-RADS 3": 1,
                "BI-RADS 4": 2,
                "BI-RADS 5": 2,
            }
        elif cancer_label_type == "binary":
            self.label_map = {
                "BI-RADS 1": 0,
                "BI-RADS 2": 0,
                "BI-RADS 3": 1,
                "BI-RADS 4": 1,
                "BI-RADS 5": 1,
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

    def load_img_to_tensor(self, image_id) -> torch.Tensor:
        """Load a grayscale image from disk and convert it to a tensor of shape (3, H, W)."""
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
                import imageio.v3 as iio

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
                image = (image - 923.4552) / 2035.7942
            else:
                image = (image - 108.3328) / 69.6431
        else:
            pass
        return image

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
                image = self.load_img_to_tensor(image_id)
                image = self.normalise_image(image, norm_kind=self.norm_kind)
                if self.transforms is not None:
                    image = self.transforms(image)
                images.append(image)

        return images, labels

    def plot(self, idx):
        images, labels = self.__getitem__(idx)
        # Transform images to the interval (0, 1)
        images = [
            (image - image.min()) / (image.max() - image.min()) for image in images
        ]
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        view = ["CC", "MLO"]
        laterality = ["Left", "Right"]
        for i in range(2):
            for j in range(2):
                axs[i, j].imshow(images[i * 2 + j].permute(1, 2, 0))
                axs[i, j].set_title(
                    f"{laterality[i]}{view[j]}, label: {labels[i].item()}"
                )
                axs[i, j].axis("off")
        plt.show()


class Patient_Cancer_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        root_folder: str,
        annotation_csv: str,
        imagefolder_path: str,
        image_format: str = "dicom",
        norm_kind: str = "dataset_zscore",
        batch_size=16,
        num_workers=8,
        train_transform=None,
        transform=None,
        cancer_label_type: str = "birads",
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
        self.train_transform = train_transform
        self.transform = transform
        self.cancer_label_type = cancer_label_type

        self.train_dataset = Patient_Cancer_Dataset(
            root_folder,
            annotation_csv,
            imagefolder_path,
            image_format=image_format,
            norm_kind=norm_kind,
            split="training",
            transform=self.train_transform,
            cancer_label_type=cancer_label_type,
        )
        self.val_dataset = Patient_Cancer_Dataset(
            root_folder,
            annotation_csv,
            imagefolder_path,
            image_format=image_format,
            norm_kind=norm_kind,
            split="validation",
            transform=self.transform,
            cancer_label_type=cancer_label_type,
        )
        self.test_dataset = Patient_Cancer_Dataset(
            root_folder,
            annotation_csv,
            imagefolder_path,
            image_format=image_format,
            norm_kind=norm_kind,
            split="test",
            transform=self.transform,
            cancer_label_type=cancer_label_type,
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

    def plot(self, idx):
        self.test_dataset.plot(idx)
