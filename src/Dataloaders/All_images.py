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
