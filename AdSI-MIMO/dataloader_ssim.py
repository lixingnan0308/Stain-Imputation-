"""
dataloader_ssim.py — PyTorch dataset for loading and preprocessing MxIF patch data.

Each sample is a multi-channel .npy file where channels correspond to different
fluorescence markers. Per-channel min-max normalization is applied on the fly so
that all pixel values lie in [0, 1] regardless of acquisition scale.
"""

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler


class MxIFReader(Dataset):
    def __init__(self, data_csv_path, split_name, marker_panel, input_markers,
                 training=False, img_size=256, percent=100):
        """
        Custom PyTorch dataset for reading and preprocessing multi-channel MxIF images.

        Args:
            data_csv_path (str): Path to a CSV file with 'Image_Paths' and 'Split_Name' columns.
            split_name (str): Data split to load ('train', 'valid', 'test', or 'inference').
            marker_panel (list): Ordered list of marker names matching the channel order in .npy files.
            input_markers (list): Subset of marker_panel to use as model input channels.
            training (bool): If True, applies augmentation; otherwise only resizes. Defaults to False.
            img_size (int): Spatial resolution to resize images to. Defaults to 256.
            percent (int): Percentage of samples to randomly include. Defaults to 100.
        """
        self.data_csv_path = data_csv_path
        self.split_name = split_name
        self.marker_panel = marker_panel
        self.input_markers = input_markers
        self.training = training
        self.img_size = img_size

        df = pd.read_csv(self.data_csv_path)
        df = df[df['Split_Name'] == split_name]
        self.x = df['Image_Paths'].tolist()

        if percent < 100:
            rand_perm = np.random.permutation(len(self.x))
            sample_count = int(len(self.x) * percent / 100)
            self.x = [self.x[i] for i in rand_perm[:sample_count]]

        # Pre-compute channel indices once to avoid repeated lookups
        self.input_channel_indexes = [
            self.marker_panel.index(m) for m in self.input_markers
        ]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Loads, normalizes, and preprocesses a single image patch.

        Args:
            idx (int): Sample index.

        Returns:
            tuple:
                input_img (torch.Tensor): Preprocessed input tensor of shape (C, H, W).
                img_path (str): Absolute path to the source .npy file.
                img_dim (list[int]): Original [height, width] before resizing (used to
                                     crop away padding in the evaluation loop).
        """
        img_path = self.x[idx]
        img = np.load(img_path)
        img = torch.tensor(img, dtype=torch.float32)

        # Per-channel min-max normalization to [0, 1]
        normalized = torch.empty_like(img)
        for i in range(img.shape[0]):
            ch = img[i]
            lo, hi = ch.min().item(), ch.max().item()
            normalized[i] = (ch - lo) / (hi - lo + 1e-8) if hi > lo else ch - lo
        img = normalized

        # Record original spatial dimensions before any resizing
        img_dim = [img.shape[1], img.shape[2]]

        input_img = img[self.input_channel_indexes, :, :]
        if self.training:
            input_img = self.preprocess_train(input_img, self.img_size)
        else:
            input_img = self.preprocess_valid(input_img, self.img_size)

        return input_img, img_path, img_dim

    @staticmethod
    def preprocess_train(img, target_size):
        """
        Applies training-time augmentation: resize, random horizontal/vertical flip,
        and a random 90-degree rotation.

        Args:
            img (torch.Tensor): Input tensor of shape (C, H, W).
            target_size (int): Output spatial resolution.

        Returns:
            torch.Tensor: Augmented tensor of shape (C, target_size, target_size).
        """
        img = F.resize(img, (target_size, target_size))
        if torch.rand(1) > 0.5:
            img = F.hflip(img)
        if torch.rand(1) > 0.5:
            img = F.vflip(img)
        img = torch.rot90(img, np.random.randint(4), [1, 2])
        return img

    @staticmethod
    def preprocess_valid(img, target_size):
        """
        Resizes the image to target_size without augmentation for validation and test.

        Args:
            img (torch.Tensor): Input tensor of shape (C, H, W).
            target_size (int): Output spatial resolution.

        Returns:
            torch.Tensor: Resized tensor of shape (C, target_size, target_size).
        """
        return F.resize(img, (target_size, target_size))

    @staticmethod
    def get_data_loader(dataset, batch_size=4, training=False, num_workers=4):
        """
        Builds a DataLoader with shuffling for training and sequential sampling otherwise.

        Args:
            dataset (MxIFReader): Dataset instance.
            batch_size (int): Number of samples per batch. Defaults to 4.
            training (bool): If True, shuffles and drops the last incomplete batch. Defaults to False.
            num_workers (int): Worker processes for data loading. Defaults to 4.

        Returns:
            DataLoader: Configured data loader.
        """
        if training:
            return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
        return DataLoader(dataset, batch_size=batch_size,
                          sampler=SequentialSampler(dataset),
                          drop_last=False, num_workers=num_workers)
