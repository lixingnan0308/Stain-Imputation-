import numpy as np
import pandas as pd

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler


class MxIFReader(Dataset):
    def __init__(self, data_csv_path, split_name, marker_panel, input_markers, output_markers,
                 training=False, img_size=256, percent=50):
        """
        Custom PyTorch Dataset for reading and preprocessing multi-channel MxIF image patches.

        Each image is stored as a .npy file with shape (C, H, W), where C is the number of
        markers in the full panel. This class selects the relevant input and output channels
        and applies per-channel min-max normalisation to [0, 1].

        Args:
            data_csv_path (str): Path to a CSV file with 'Image_Paths' and 'Split_Name' columns.
            split_name (str): Split to load, e.g. 'train', 'valid', or 'test'.
            marker_panel (list): Marker names in the same channel order as the .npy files.
            input_markers (list): Subset of marker_panel to use as model input channels.
            output_markers (list): Subset of marker_panel to use as model output (target) channels.
            training (bool): If True, applies augmentation; otherwise applies deterministic resize.
                             Defaults to False.
            img_size (int): Spatial size (height = width) to resize patches to. Defaults to 256.
            percent (int): Percentage of training samples to include (for data-efficient experiments).
                           Defaults to 50.
        """
        self.data_csv_path = data_csv_path
        self.split_name = split_name
        self.marker_panel = marker_panel
        self.input_markers = input_markers
        self.output_markers = output_markers
        self.training = training
        self.img_size = img_size

        # Load image paths for the requested split
        df = pd.read_csv(self.data_csv_path)
        df = df[df['Split_Name'] == split_name]
        self.x = df['Image_Paths'].tolist()

        # Optionally sub-sample the training set
        if percent < 100:
            rand_perm = np.random.permutation(len(self.x))
            sample_count = int(len(self.x) * percent / 100)
            self.x = [self.x[idx] for idx in rand_perm[:sample_count]]

        # Pre-compute channel indices to avoid repeated lookups in __getitem__
        self.input_channel_indexes = [marker_panel.index(m) for m in input_markers]
        self.output_channel_indexes = [marker_panel.index(m) for m in output_markers]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Loads, normalises, augments, and slices a single image patch.

        The original spatial dimensions are returned alongside the processed patch so that
        evaluation code can crop away any padding introduced by resizing.

        Args:
            idx (int): Sample index.

        Returns:
            tuple:
                input_img (torch.Tensor): Input channels, shape (len(input_markers), H, W).
                output_img (torch.Tensor): Output (target) channels, shape (len(output_markers), H, W).
                img_path (str): Absolute path to the source .npy file.
                img_dim (list[int]): Original [height, width] before resizing.
        """
        img_path = self.x[idx]
        img = np.load(img_path)  # shape: (C, H, W)
        img = torch.tensor(img, dtype=torch.float32)

        # Per-channel min-max normalisation to [0, 1]
        minmax_normalized = torch.empty_like(img)
        for i in range(img.shape[0]):
            channel = img[i]
            min_val = channel.min().item()
            max_val = channel.max().item()
            if max_val != min_val:
                minmax_normalized[i] = (channel - min_val) / (max_val - min_val)
            else:
                # Constant channel: subtract min so the output is all zeros
                minmax_normalized[i] = channel - min_val
        img = minmax_normalized

        # Record original spatial dimensions before any resizing
        img_dim = [img.shape[1], img.shape[2]]

        if self.training:
            img = self.preprocess_train(img, self.img_size)
        else:
            img = self.preprocess_valid(img, self.img_size)

        # Select the requested input and output channels
        input_img = torch.stack([img[c] for c in self.input_channel_indexes])
        output_img = torch.stack([img[c] for c in self.output_channel_indexes])

        return input_img, output_img, img_path, img_dim

    @staticmethod
    def preprocess_train(img, target_size):
        """
        Applies training augmentations: resize, random horizontal flip, random vertical
        flip, and random 90-degree rotation.

        Args:
            img (torch.Tensor): Image tensor of shape (C, H, W).
            target_size (int): Target spatial size (applied to both height and width).

        Returns:
            torch.Tensor: Augmented image tensor of shape (C, target_size, target_size).
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
        Applies deterministic preprocessing for validation and inference: resize only.

        Args:
            img (torch.Tensor): Image tensor of shape (C, H, W).
            target_size (int): Target spatial size (applied to both height and width).

        Returns:
            torch.Tensor: Resized image tensor of shape (C, target_size, target_size).
        """
        img = F.resize(img, (target_size, target_size))
        return img

    @staticmethod
    def get_data_loader(dataset, batch_size=4, training=False, num_workers=4):
        """
        Wraps a MxIFReader dataset in a PyTorch DataLoader.

        Training loaders shuffle data and drop the last incomplete batch.
        Validation/test loaders use a sequential sampler to preserve sample order.

        Args:
            dataset (MxIFReader): Dataset instance to wrap.
            batch_size (int): Number of samples per batch. Defaults to 4.
            training (bool): If True, creates a shuffled training loader. Defaults to False.
            num_workers (int): Number of worker processes for parallel data loading. Defaults to 4.

        Returns:
            DataLoader: Configured data loader.
        """
        if training:
            return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=num_workers)
        else:
            return DataLoader(dataset, batch_size=batch_size,
                              sampler=SequentialSampler(dataset),
                              drop_last=False, num_workers=num_workers)
