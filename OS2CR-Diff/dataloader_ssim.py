import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, SequentialSampler

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)


class MxIFReader(Dataset):
    def __init__(self, data_csv_path, split_name, marker_panel,
                 input_markers, training=False, img_size=256, percent=50):
        """
        PyTorch dataset for multi-channel multiplexed immunofluorescence (MxIF) images.

        Each image is stored as a .npy file of shape (C, H, W).  The last channel
        appended by this dataset is always treated as the reconstruction target
        (real CD8); the preceding C-1 channels correspond to the marker panel.

        Args:
            data_csv_path (str):   Path to CSV file with 'Image_Paths' and 'Split' columns.
            split_name    (str):   Data split to load ('train', 'valid', 'test', …).
            marker_panel  (list):  Ordered marker names matching the .npy channel order.
            input_markers (list):  Marker names to use as model input channels.
            training      (bool):  If True, applies data augmentation. Defaults to False.
            img_size      (int):   Spatial resolution returned by the loader. Defaults to 256.
            percent       (int):   Percentage of samples to include (1–100). Defaults to 50.
        """
        self.data_csv_path = data_csv_path
        self.split_name    = split_name
        self.marker_panel  = marker_panel
        self.input_markers = input_markers
        self.training      = training
        self.img_size      = img_size

        df      = pd.read_csv(self.data_csv_path)
        df      = df[df['Split'] == split_name]
        self.x  = df['Image_Paths'].tolist()

        if percent < 100:
            rand_perm    = np.random.permutation(len(self.x))
            sample_count = int(len(self.x) * percent / 100)
            self.x       = [self.x[i] for i in rand_perm[:sample_count]]

        self.input_channel_indexes = [
            self.marker_panel.index(m) for m in self.input_markers
        ]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        Returns a tuple (input_img, img_path, img_dim).

        input_img (torch.Tensor): Shape (len(input_markers) + 1, img_size, img_size),
                                  values in [-1, 1].  The +1 channel is the real CD8
                                  target (last channel of the .npy file).
        img_path  (str):          Full path to the source .npy file.
        img_dim   (list[int]):    Original spatial dimensions [H, W] before resizing.
        """
        img_path = self.x[idx]
        img = torch.tensor(np.load(img_path), dtype=torch.float32)
        img = img * 2 - 1  # normalise to [-1, 1]

        img_dim = [img.shape[1], img.shape[2]]

        img_t = self.preprocess_train(img, self.img_size) if self.training \
            else self.preprocess_valid(img, self.img_size)

        # Build output tensor: all input channels + real CD8 (last channel).
        n_channels = len(self.input_channel_indexes) + 1
        input_img = torch.zeros((n_channels, self.img_size, self.img_size))
        for i in range(n_channels):
            input_img[i] = img_t[i]

        return input_img, img_path, img_dim

    @staticmethod
    def preprocess_train(img, target_size):
        """
        Applies training-time augmentation: resize, random H/V flip, random 90° rotation.

        Args:
            img         (torch.Tensor): Input tensor of shape (C, H, W).
            target_size (int):          Target spatial size.

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
        Applies validation/test-time preprocessing: resize only.

        Args:
            img         (torch.Tensor): Input tensor of shape (C, H, W).
            target_size (int):          Target spatial size.

        Returns:
            torch.Tensor: Resized tensor of shape (C, target_size, target_size).
        """
        return F.resize(img, (target_size, target_size))

    @staticmethod
    def get_data_loader(dataset, batch_size=4, training=False, num_workers=4):
        """
        Builds a DataLoader for the given MxIFReader dataset.

        Args:
            dataset     (MxIFReader): Dataset instance.
            batch_size  (int):        Batch size. Defaults to 4.
            training    (bool):       Shuffle and drop last batch when True.
                                      Defaults to False.
            num_workers (int):        Number of data-loading worker processes.
                                      Defaults to 4.

        Returns:
            torch.utils.data.DataLoader
        """
        if training:
            return DataLoader(dataset, batch_size=batch_size,
                              shuffle=True, drop_last=True, num_workers=num_workers)
        return DataLoader(dataset, batch_size=batch_size,
                          sampler=SequentialSampler(dataset),
                          drop_last=False, num_workers=num_workers)
