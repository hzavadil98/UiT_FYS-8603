import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as T
from pydicom import dcmread
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class Synthetic_2v_Dataset(Dataset):
    def __init__(self, n_samples, img_size=512, transform=None):
        self.n_samples = n_samples
        self.transform = transform
        self.img_size = img_size

        self.labels = np.random.randint(0, 3, size=(self.n_samples, 2))

        self.parameters = [
            self.generate_random_parameters(task1, task2)
            for task1, task2 in self.labels
        ]

    def generate_sinusoidal_pattern(self, z, img_size):
        fx = z[0]  # frequency x in [0, 10]
        fy = z[1]  # frequency y
        phi = (2 * np.pi * z[2]) - np.pi  # phase in [-π, π]

        x = np.linspace(-np.pi, np.pi, img_size)
        y = np.linspace(-np.pi, np.pi, img_size)
        X, Y = np.meshgrid(x, y)

        wave = np.sin(fx * X + fy * Y + phi)

        # Apply exponential decay
        decay = np.exp(-((X) ** 2 + (Y) ** 2) / (2 * (np.pi / 2) ** 2))
        wave *= decay

        # flip the image vertically with 50% chance
        if np.random.rand() > 0.5:
            wave = np.flipud(wave)

        image = ((wave - wave.min()) / (wave.max() - wave.min()) * 255).astype(np.uint8)
        return image

    def generate_sinc_pattern(self, z, img_size):
        a = z[0]  # main axis in [0, 2]
        b = z[1]  # minor axis in [0, 2]
        theta = np.pi * z[2]  # rotation angle in [0, π]

        x = np.linspace(-10, 10, img_size)
        y = np.linspace(-10, 10, img_size)
        X, Y = np.meshgrid(x, y)

        # Rotation transformation
        X_rot = X * np.cos(theta) - Y * np.sin(theta)
        Y_rot = X * np.sin(theta) + Y * np.cos(theta)

        sinc_wave = np.sinc((X_rot / a) ** 2 + (Y_rot / b) ** 2)

        image = (
            (sinc_wave - sinc_wave.min()) / (sinc_wave.max() - sinc_wave.min()) * 255
        ).astype(np.uint8)
        return image

    def generate_random_parameters(self, task1, task2):
        match task1:
            case 0:
                task1_indiv1 = [[0, 2], [7, 8]]
                task1_indiv2 = [[11, 12], [0, 2]]
                task1_shared = [[1, 3], [1, 3]]
            case 1:
                task1_indiv1 = [[0, 2], [9, 10]]
                task1_indiv2 = [[9, 10], [0, 2]]
                task1_shared = [[4, 6], [4, 6]]
            case 2:
                task1_indiv1 = [[0, 2], [11, 12]]
                task1_indiv2 = [[7, 8], [0, 2]]
                task1_shared = [[7, 9], [7, 9]]
            case _:
                print("Wrong class label")

        match task2:
            case 0:
                task2_indiv1 = [[1, 2.5], 2, [30 / 180, 60 / 180]]
                task2_indiv2 = [[3.5, 5], 4, [120 / 180, 150 / 180]]
                task2_shared = [[1, 5], 2, [-15 / 180, 15 / 180]]
            case 1:
                task2_indiv1 = [[1, 2.5], 2.5, [30 / 180, 60 / 180]]
                task2_indiv2 = [[3.5, 5], 3, [120 / 180, 150 / 180]]
                task2_shared = [[1, 5], 2.5, [-15 / 180, 15 / 180]]
            case 2:
                task2_indiv1 = [[1, 2.5], 3, [30 / 180, 60 / 180]]
                task2_indiv2 = [[3.5, 5], 2, [120 / 180, 150 / 180]]
                task2_shared = [[1, 5], 3, [-15 / 180, 15 / 180]]
            case _:
                print("Wrong class label")

        task1_indiv1 = [
            random.uniform(task1_indiv1[0][0], task1_indiv1[0][1]),
            random.uniform(task1_indiv1[1][0], task1_indiv1[1][1]),
            random.random(),
        ]
        task1_indiv2 = [
            random.uniform(task1_indiv2[0][0], task1_indiv2[0][1]),
            random.uniform(task1_indiv2[1][0], task1_indiv2[1][1]),
            random.random(),
        ]
        task1_shared = [
            random.uniform(task1_shared[0][0], task1_shared[0][1]),
            random.uniform(task1_shared[1][0], task1_shared[1][1]),
            random.random(),
        ]

        task2_indiv1 = random.uniform(
            task2_indiv1[0][0], task2_indiv1[0][1]
        ) * np.array([1, task2_indiv1[1], 0]) + np.array(
            [0, 0, random.uniform(task2_indiv1[2][0], task2_indiv1[2][1])]
        )
        task2_indiv2 = random.uniform(
            task2_indiv2[0][0], task2_indiv2[0][1]
        ) * np.array([1, task2_indiv2[1], 0]) + np.array(
            [0, 0, random.uniform(task2_indiv2[2][0], task2_indiv2[2][1])]
        )
        task2_shared = random.uniform(
            task2_shared[0][0], task2_shared[0][1]
        ) * np.array([1, task2_shared[1], 0]) + np.array(
            [0, 0, random.uniform(task2_shared[2][0], task2_shared[2][1])]
        )

        return [
            task1_indiv1,
            task1_indiv2,
            task1_shared,
            task2_indiv1,
            task2_indiv2,
            task2_shared,
        ]

    def generate_image(self, index):
        sub_img_size = self.img_size // 3 + 1

        (
            task1_indiv1,
            task1_indiv2,
            task1_shared,
            task2_indiv1,
            task2_indiv2,
            task2_shared,
        ) = self.parameters[index]

        task1_indiv1 = self.generate_sinusoidal_pattern(
            task1_indiv1, img_size=sub_img_size
        )
        task1_indiv2 = self.generate_sinusoidal_pattern(
            task1_indiv2, img_size=sub_img_size
        )
        task1_shared = self.generate_sinusoidal_pattern(
            task1_shared, img_size=sub_img_size
        )

        task2_indiv1 = self.generate_sinc_pattern(task2_indiv1, img_size=sub_img_size)
        task2_indiv2 = self.generate_sinc_pattern(task2_indiv2, img_size=sub_img_size)
        task2_shared = self.generate_sinc_pattern(task2_shared, img_size=sub_img_size)

        fill = np.zeros((sub_img_size, sub_img_size), dtype=np.uint8)

        view1 = np.block(
            [
                [task1_shared, task1_indiv1, fill],
                [task1_indiv1, fill, task2_indiv1],
                [fill, task2_indiv1, task2_shared],
            ]
        )

        view2 = np.block(
            [
                [task1_indiv2, task1_shared, task1_indiv2],
                [fill, fill, fill],
                [task2_indiv2, task2_shared, task2_indiv2],
            ]
        )

        # Convert views to float32 and stack
        images = np.stack([view1, view2], axis=0).astype(np.float32)

        # Convert to torch tensor and add channel dimension if needed
        tensor_images = torch.from_numpy(images).unsqueeze(1)  # shape: (2, 1, H, W)

        # Concatenate local transform with self.transform if provided
        # Resize using torchvision transforms
        transform = T.Compose(
            [
                T.Resize((self.img_size, self.img_size)),  # expects (C, H, W)
                T.Normalize(mean=[0.0], std=[255.0]),  # normalize to [0,1]
            ]
        )

        if self.transform is not None:
            transform = T.Compose([*transform.transforms, *self.transform.transforms])
        # Apply transform to each view
        tensor_images = torch.stack(
            [transform(img) for img in tensor_images], dim=0
        )  # shape: (2, 1, img_size, img_size)

        return tensor_images

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.generate_image(idx), self.labels[idx]

    def plot(self, idx):
        images, label = self.__getitem__(idx)
        # Transform images to the interval (0, 1)
        images = [
            (image - image.min()) / (image.max() - image.min()) for image in images
        ]
        fig, axs = plt.subplots(1, 2)

        for i in range(2):
            # axs[i].imshow(images[i].permute(1, 2, 0))
            axs[i].imshow(images[i][0], cmap="gray")
            axs[i].axis("off")
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_aspect("equal")
            axs[i].set_title(f"Label: {label[i]}")
        plt.show()


class Synthetic_2v_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        n_samples=[10000, 2000, 2000],
        img_size=512,
        batch_size=32,
        num_workers=4,
        transform=None,
    ):
        super().__init__()
        self.n_samples = n_samples
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

        self.train_dataset = Synthetic_2v_Dataset(
            n_samples=self.n_samples[0],
            img_size=self.img_size,
            transform=self.transform,
        )
        self.val_dataset = Synthetic_2v_Dataset(
            n_samples=self.n_samples[1],
            img_size=self.img_size,
            transform=self.transform,
        )
        self.test_dataset = Synthetic_2v_Dataset(
            n_samples=self.n_samples[2],
            img_size=self.img_size,
            transform=self.transform,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def main():
    # Set the random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    transform = T.Compose(
        [
            T.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Create a DataLoader
    dataloader = Synthetic_2v_Dataloader(
        n_samples=[1000, 200, 200],
        img_size=512,
        batch_size=16,
        num_workers=4,
        transform=transform,
    )

    # Check if the dataloader passes the model
    batch = next(iter(dataloader.train_dataloader()))
    dataset = dataloader.train_dataset
    images, labels = batch
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
    print(torch.min(images), torch.max(images))

    # Plot an example
    dataset.plot(0)


if __name__ == "__main__":
    main()
