import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.v2 as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Synthetic_2v_Dataset(Dataset):
    def __init__(self, n_samples, image_save_dir, img_size=512, transform=None):
        self.n_samples = n_samples
        self.transform = transform
        self.img_size = img_size
        self.image_save_dir = pathlib.Path(image_save_dir)

        # check if the directory exists, if not create it
        if not self.image_save_dir.exists():
            self.create_dataset()
        else:
            self.load_dataset()

    def load_dataset(self):
        """
        Load labels, parameters and image paths created by create_dataset.
        """
        self.labels = np.load(self.image_save_dir / "labels.npy")
        self.parameters = np.load(
            self.image_save_dir / "parameters.npy", allow_pickle=True
        )
        self.image_paths = np.load(
            self.image_save_dir / "image_paths.npy", allow_pickle=True
        )

        # Convert stored paths (strings) to pathlib.Path objects for use later
        self.image_paths = [
            (pathlib.Path(p[0]), pathlib.Path(p[1])) for p in self.image_paths
        ]

    def create_dataset(self):
        """
        Create the dataset by generating images and saving them to disk as PNGs.
        """
        # Create the save directory if it doesn't exist
        self.image_save_dir.mkdir(parents=True, exist_ok=True)

        self.labels = np.random.randint(0, 3, size=(self.n_samples, 2))
        np.save(self.image_save_dir / "labels.npy", self.labels)

        # Generate and save parameters
        self.parameters = [
            self.generate_random_parameters(task1, task2)
            for task1, task2 in self.labels
        ]
        np.save(self.image_save_dir / "parameters.npy", self.parameters)

        # Generate images and save as PNG
        self.image_paths = []
        progress_bar = tqdm(range(self.n_samples), desc="Generating synthetic images")
        for i in progress_bar:
            view0_path = self.image_save_dir / f"image_{i}_view_0.png"
            view1_path = self.image_save_dir / f"image_{i}_view_1.png"

            pil_images = self.generate_image(i)

            pil_images[0].save(view0_path)
            pil_images[1].save(view1_path)

            # Store string paths so numpy can save/load reliably
            self.image_paths.append((str(view0_path), str(view1_path)))

        np.save(self.image_save_dir / "image_paths.npy", self.image_paths)

        self.image_paths = [
            (pathlib.Path(p[0]), pathlib.Path(p[1])) for p in self.image_paths
        ]

    def generate_sinusoidal_pattern(self, z, img_size):
        fx = z[0]  # frequency x
        fy = z[1]  # frequency y
        phi = (2 * np.pi * z[2]) - np.pi

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
        a = z[0]
        b = z[1]
        theta = np.pi * z[2]

        x = np.linspace(-10, 10, img_size)
        y = np.linspace(-10, 10, img_size)
        X, Y = np.meshgrid(x, y)

        X_rot = X * np.cos(theta) - Y * np.sin(theta)
        Y_rot = X * np.sin(theta) + Y * np.cos(theta)

        sinc_wave = np.sinc((X_rot / a) ** 2 + (Y_rot / b) ** 2)

        image = (
            (sinc_wave - sinc_wave.min()) / (sinc_wave.max() - sinc_wave.min()) * 255
        ).astype(np.uint8)
        return image

    def generate_random_parameters(self, task1, task2):
        """
        Return the random parameters used to build the two views. Kept the
        same logic as before so dataset semantics don't change.
        """
        match task1:
            case 0:
                task1_indiv1 = [1, 7.5]
                task1_indiv2 = [9.5, 1]
                task1_shared = [2, 2]
            case 1:
                task1_indiv1 = [1, 9.5]
                task1_indiv2 = [7.5, 1]
                task1_shared = [8, 8]
            case 2:
                task1_indiv1 = [1, 11.5]
                task1_indiv2 = [11.5, 1]
                task1_shared = [5, 5]
            case _:
                print("Wrong class label")

        match task2:
            case 0:
                task2_indiv1 = [[1, 2.5], 2, [30 / 180, 60 / 180]]
                task2_indiv2 = [[3.5, 5], 3, [120 / 180, 150 / 180]]
                task2_shared = [[1, 5], 2, [-15 / 180, 15 / 180]]
            case 1:
                task2_indiv1 = [[1, 2.5], 2.5, [30 / 180, 60 / 180]]
                task2_indiv2 = [[3.5, 5], 4, [120 / 180, 150 / 180]]
                task2_shared = [[1, 5], 3, [-15 / 180, 15 / 180]]
            case 2:
                task2_indiv1 = [[1, 2.5], 3, [30 / 180, 60 / 180]]
                task2_indiv2 = [[3.5, 5], 2, [120 / 180, 150 / 180]]
                task2_shared = [[1, 5], 2.5, [-15 / 180, 15 / 180]]
            case _:
                print("Wrong class label")

        match task1:
            case 0:
                match task2:
                    case 0:
                        task_shared = [0.5, 0.5]
                    case 1:
                        task_shared = [2, 2]
                    case 2:
                        task_shared = [3.5, 3.5]
                    case _:
                        print("Wrong class label")
            case 1:
                match task2:
                    case 0:
                        task_shared = [5, 5]
                    case 1:
                        task_shared = [6.5, 6.5]
                    case 2:
                        task_shared = [8, 8]
                    case _:
                        print("Wrong class label")
            case 2:
                match task2:
                    case 0:
                        task_shared = [9.5, 9.5]
                    case 1:
                        task_shared = [11, 11]
                    case 2:
                        task_shared = [12.5, 12.5]
                    case _:
                        print("Wrong class label")
            case _:
                print("Wrong class label")

        sigma = 2
        task1_indiv1 = [
            np.random.normal(loc=task1_indiv1[0], scale=sigma),
            np.random.normal(loc=task1_indiv1[1], scale=sigma),
            np.random.uniform(0, 1),
        ]
        task1_indiv2 = [
            np.random.normal(loc=task1_indiv2[0], scale=sigma),
            np.random.normal(loc=task1_indiv2[1], scale=sigma),
            np.random.uniform(0, 1),
        ]
        task1_shared = [
            np.random.normal(loc=task1_shared[0], scale=1.5 * sigma),
            np.random.normal(loc=task1_shared[1], scale=1.5 * sigma),
            np.random.uniform(0, 1),
        ]

        sigma = 0.42
        task2_indiv1 = random.uniform(
            task2_indiv1[0][0], task2_indiv1[0][1]
        ) * np.array(
            [1, np.random.normal(loc=task2_indiv1[1], scale=sigma), 0]
        ) + np.array([0, 0, random.uniform(task2_indiv1[2][0], task2_indiv1[2][1])])
        task2_indiv2 = random.uniform(
            task2_indiv2[0][0], task2_indiv2[0][1]
        ) * np.array(
            [1, np.random.normal(loc=task2_indiv2[1], scale=2 * sigma), 0]
        ) + np.array([0, 0, random.uniform(task2_indiv2[2][0], task2_indiv2[2][1])])
        task2_shared = random.uniform(
            task2_shared[0][0], task2_shared[0][1]
        ) * np.array(
            [1, np.random.normal(loc=task2_shared[1], scale=sigma), 0]
        ) + np.array([0, 0, random.uniform(task2_shared[2][0], task2_shared[2][1])])

        sigma = 1.5
        task_shared = [
            np.random.normal(loc=task_shared[0], scale=sigma),
            np.random.normal(loc=task_shared[1], scale=sigma),
            np.random.uniform(0, 1),
        ]

        return [
            task1_indiv1,
            task1_indiv2,
            task1_shared,
            task2_indiv1,
            task2_indiv2,
            task2_shared,
            task_shared,
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
            task_shared,
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

        task_shared = self.generate_sinusoidal_pattern(
            task_shared, img_size=sub_img_size
        )

        fill = np.zeros((sub_img_size, sub_img_size), dtype=np.uint8)

        view1 = np.block(
            [
                [task1_shared, task1_indiv1, fill],
                [task1_indiv1, task_shared, task2_indiv1],
                [fill, task2_indiv1, task2_shared],
            ]
        )

        view2 = np.block(
            [
                [task1_indiv2, task1_shared, task1_indiv2],
                [fill, task_shared, fill],
                [task2_indiv2, task2_shared, task2_indiv2],
            ]
        )

        # Convert uint8 arrays to PIL images and resize to target size
        pil1 = (
            Image.fromarray(view1)
            .convert("L")
            .resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        )
        pil2 = (
            Image.fromarray(view2)
            .convert("L")
            .resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        )

        return [pil1, pil2]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        view0_path, view1_path = self.image_paths[idx]

        # Load PNGs as grayscale PIL images and convert to float tensors [0,1]
        pil0 = Image.open(view0_path).convert("L")
        pil1 = Image.open(view1_path).convert("L")

        img0 = torch.from_numpy(np.array(pil0).astype(np.float32) / 255.0).unsqueeze(0)
        img1 = torch.from_numpy(np.array(pil1).astype(np.float32) / 255.0).unsqueeze(0)

        if self.transform is not None:
            images = [self.transform(img0), self.transform(img1)]
        else:
            images = [img0, img1]

        return images, self.labels[idx, 0], self.labels[idx, 1]

    def plot(self, idx):
        images, label1, label2 = self.__getitem__(idx)
        # Transform images to the interval (0, 1)
        images = [
            (image - image.min()) / (image.max() - image.min()) for image in images
        ]
        fig, axs = plt.subplots(1, 2)

        for i in range(2):
            axs[i].imshow(images[i][0], cmap="gray")
            axs[i].axis("off")
            axs[i].set_xticks([])
            axs[i].set_yticks([])
            axs[i].set_aspect("equal")
            axs[i].set_title(f"Label 1: {label1}, Label 2: {label2}")
        plt.show()


class Synthetic_2v_Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        n_samples=[10000, 2000, 2000],
        img_size=512,
        batch_size=32,
        num_workers=4,
        train_transform=None,
        transform=None,
        image_save_dir="synthetic_images",
    ):
        super().__init__()
        self.n_samples = n_samples
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.transform = transform
        self.image_save_dir = pathlib.Path(image_save_dir)

        self.train_dataset = Synthetic_2v_Dataset(
            n_samples=self.n_samples[0],
            img_size=self.img_size,
            transform=self.train_transform,
            image_save_dir=self.image_save_dir / "train",
        )
        self.val_dataset = Synthetic_2v_Dataset(
            n_samples=self.n_samples[1],
            img_size=self.img_size,
            transform=self.transform,
            image_save_dir=self.image_save_dir / "val",
        )
        self.test_dataset = Synthetic_2v_Dataset(
            n_samples=self.n_samples[2],
            img_size=self.img_size,
            transform=self.transform,
            image_save_dir=self.image_save_dir / "test",
        )

    def train_dataloader(self, shuffle=True):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
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
        n_samples=[100, 50, 50],
        img_size=512,
        batch_size=16,
        num_workers=4,
        transform=transform,
    )

    # Check if the dataloader passes the model
    batch = next(iter(dataloader.train_dataloader()))
    dataset = dataloader.train_dataset
    images, y1, y2 = batch
    print(f"Batch shape: {images[0].shape}, Labels: {y1}, {y2}")
    print(torch.min(images[0]), torch.max(images[0]))

    # Plot an example
    dataset.plot(0)


if __name__ == "__main__":
    main()
