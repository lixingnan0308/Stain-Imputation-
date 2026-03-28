import json
import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients
from dataloader import MxIFReader

from scipy import stats as st
from skimage.metrics import structural_similarity as ssim
import platform


class Trainer:
    def __init__(self,
                 marker_panel,
                 fixed_markers,
                 potential_output_markers,
                 results_dir,
                 target_marker,
                 lr=0.002, seed=1):
        """
        Base trainer class for MxIF stain imputation models.

        Args:
            marker_panel (list): Marker names in channel order for the MxIF images.
            fixed_markers (list): Markers always present as input (e.g., DAPI, autofluorescence).
            potential_output_markers (list): Markers that can serve as imputation targets.
            results_dir (str): Directory for saving checkpoints and training statistics.
            target_marker (list): Specific marker(s) to impute in this training run.
            lr (float): Learning rate for the Adam optimizer. Defaults to 0.002.
            seed (int): Random seed for reproducibility. Defaults to 1.
        """
        self.marker_panel = marker_panel
        self.target_marker = target_marker
        self.fixed_markers = fixed_markers
        self.pot_output_markers = potential_output_markers
        self.results_dir = results_dir
        self.lr = lr
        self.seed = seed

        self.counter = 0
        self.lowest_loss = np.Inf
        # Select the appropriate compute device based on platform
        if platform.system() == 'Darwin':
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.train_dataset = None
        self.valid_dataset = None
        self.train_loader = None
        self.valid_loader = None

        self.model_g = None
        self.optimizer = None
        self.loss_l1 = None
        self.loss_mse = None
        self.stain_indexes = []

        os.makedirs(self.results_dir, exist_ok=True)

    def set_seed(self, seed):
        """
        Sets the random seed across all libraries for reproducibility.

        Args:
            seed (int): Random seed.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # macOS requires explicit deterministic mode for MPS reproducibility
        if platform.system() == 'Darwin':
            torch.use_deterministic_algorithms(True)

    def init_data_loader(self, data_csv_path, percent=100,
                         img_size=256, batch_size=64,
                         num_workers=4,
                         input_marker=[],
                         output_marker=[]):
        """
        Initializes the data loaders for training and validation splits.

        Args:
            data_csv_path (str): Path to the CSV file listing image paths and split assignments.
            percent (int): Percentage of training data to use. Defaults to 100.
            img_size (int): Spatial size to which image patches are resized. Defaults to 256.
            batch_size (int): Batch size for the data loader. Defaults to 64.
            num_workers (int): Number of worker processes for data loading. Defaults to 4.
            input_marker (list): Channel names to use as model input.
            output_marker (list): Channel names to use as model output (imputation target).

        Returns:
            list: [train_dataset, train_loader, valid_dataset, valid_loader]
        """
        train_dataset = MxIFReader(data_csv_path=data_csv_path, split_name='train',
                                   marker_panel=self.marker_panel,
                                   input_markers=input_marker,
                                   output_markers=output_marker,
                                   training=True, img_size=img_size, percent=percent)
        train_loader = MxIFReader.get_data_loader(train_dataset, batch_size=batch_size,
                                                  training=True, num_workers=num_workers)

        valid_dataset = MxIFReader(data_csv_path=data_csv_path, split_name='valid',
                                   marker_panel=self.marker_panel,
                                   input_markers=input_marker,
                                   output_markers=output_marker,
                                   training=False, img_size=img_size)
        valid_loader = MxIFReader.get_data_loader(valid_dataset, batch_size=batch_size,
                                                  training=False, num_workers=num_workers)

        return [train_dataset, train_loader, valid_dataset, valid_loader]

    def init_model(self, is_train=False, input_marker=[], output_marker=[], had_d=False):
        """
        Initializes the generator model. Intended to be overridden by subclasses.

        Args:
            is_train (bool): If True, moves the model to the selected device. Defaults to False.
            input_marker (list): Input channel names (used to set in_channels).
            output_marker (list): Output channel names (used to set out_channels).
            had_d (bool): Whether a discriminator already exists; used by subclasses.
        """
        self.model_g = Generator(in_channels=len(input_marker),
                                 out_channels=len(output_marker),
                                 init_features=32)
        self.model_g = self.model_g.apply(weights_init)
        if is_train:
            self.model_g = self.model_g.to(device=self.device)

    def init_optimizer(self):
        """Initializes the Adam optimizer for the generator."""
        self.optimizer = optim.Adam(self.model_g.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def init_loss_function(self):
        """Initializes L1 and MSE loss functions used during training."""
        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()

    def train(self, data_csv_path, percent=100, img_size=256, batch_size=64, num_workers=4,
              max_epochs=200, minimum_epochs=50, patience=25, load_model_ckpt=False):
        """
        Main training loop with early stopping based on a combined Pearson + SSIM metric.

        Sets up one data loader and generator branch per target marker, then trains for
        up to max_epochs, saving checkpoints whenever the validation metric improves or
        every 10 epochs as a fallback.

        Args:
            data_csv_path (str): Path to the CSV file with image paths and split labels.
            percent (int): Percentage of training data to use. Defaults to 100.
            img_size (int): Spatial resolution of image patches. Defaults to 256.
            batch_size (int): Batch size. Defaults to 64.
            num_workers (int): Number of data-loading worker processes. Defaults to 4.
            max_epochs (int): Maximum number of training epochs. Defaults to 200.
            minimum_epochs (int): Minimum epochs before early stopping is considered. Defaults to 50.
            patience (int): Epochs without improvement before early stopping triggers. Defaults to 25.
            load_model_ckpt (bool): Whether to resume from a saved checkpoint. Defaults to False.

        Returns:
            dict: Training history with keys for train/valid loss, L1, MSE, and combined metric.
        """
        self.counter = 0
        self.lowest_loss = -np.Inf
        self.set_seed(seed=self.seed)
        self.branch_loaders = []
        self.branch_models_g = []
        self.optimizers = []
        self.losses = []
        self.has_discriminator = False
        self.has_discriminator_optim = False
        self.load_model_ckpt = load_model_ckpt
        self.stain_indexes = []

        # Build one branch (data loader + generator) per target marker
        for index, potential_output in enumerate(self.target_marker):
            # Remove the current target from the pool of potential outputs so it
            # is not treated as an available input in subsequent branches
            self.pot_output_markers.remove(potential_output)

            # Input = full marker panel minus the marker being imputed
            input_marker = self.marker_panel.copy()
            input_marker.remove(potential_output)

            output_marker = potential_output

            # Identify the non-fixed (stainable) input channels; these are the ones
            # that may be randomly masked during training for curriculum augmentation
            left_pot_marker = input_marker.copy()
            for marker in self.fixed_markers:
                left_pot_marker.remove(marker)

            # Also exclude other potential output markers from the current input
            for marker in self.pot_output_markers:
                input_marker.remove(marker)

            # Channel indices of the maskable (non-fixed) input markers
            stain_index = [input_marker.index(element)
                           for element in input_marker if element in left_pot_marker]
            self.stain_indexes.append(stain_index)

            branch_loader = self.init_data_loader(data_csv_path,
                                                  percent=percent,
                                                  img_size=img_size,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  input_marker=input_marker,
                                                  output_marker=[output_marker])
            self.branch_loaders.append(branch_loader)

            # The discriminator is shared across all branches; create it only once
            if not self.has_discriminator:
                branch_model_g, self.model_d = self.init_model(
                    is_train=True,
                    input_marker=input_marker,
                    output_marker=[output_marker],
                    had_d=self.has_discriminator
                )
                if self.load_model_ckpt:
                    branch_model_g = self.load_model(
                        ckpt_path=os.path.join(self.results_dir, 'checkpoint_65.pt'),
                        model_g=branch_model_g, index=index
                    )
                    branch_model_g = branch_model_g.to(device=self.device)
                    self.load_model_d(ckpt_path=os.path.join(self.results_dir, 'checkpoint_d_65.pt'))

                self.branch_models_g.append(branch_model_g)
                self.model_d = self.model_d.to(device=self.device)
                self.has_discriminator = True
            else:
                branch_model_g = self.init_model(
                    is_train=True,
                    input_marker=input_marker,
                    output_marker=[output_marker],
                    had_d=self.has_discriminator
                )
                if self.load_model_ckpt:
                    branch_model_g = self.load_model(
                        ckpt_path=os.path.join(self.results_dir, 'checkpoint_65.pt'),
                        model_g=branch_model_g, index=index
                    )
                    branch_model_g = branch_model_g.to(device=self.device)
                    self.load_model_d(ckpt_path=os.path.join(self.results_dir, 'checkpoint_d_65.pt'))
                self.branch_models_g.append(branch_model_g)

            # The discriminator optimizer is also shared; create it only on the first branch
            if not self.has_discriminator_optim:
                optimizer, self.optimizer_d = self.init_optimizer(
                    model_g=branch_model_g,
                    model_d=self.model_d,
                    has_o=self.has_discriminator_optim
                )
                self.optimizers.append(optimizer)
                self.has_discriminator_optim = True
            else:
                optimizer = self.init_optimizer(
                    model_g=branch_model_g,
                    model_d=self.model_d,
                    has_o=self.has_discriminator_optim
                )
                self.optimizers.append(optimizer)

        self.init_loss_function()

        result_dict = {
            'train_loss': [], 'valid_loss': [],
            'train_l1': [], 'valid_l1': [],
            'train_mse': [], 'valid_mse': [],
            'valid': []
        }

        for epoch in range(max_epochs):
            start_time = time.time()

            train_loss, train_l1, train_mse = [], [], []
            for index, loader in enumerate(self.branch_loaders):
                train_loss_b, train_l1_b, train_mse_b = self.train_loop(loader[1], index, epoch)
                train_loss.append(train_loss_b)
                train_l1.append(train_l1_b)
                train_mse.append(train_mse_b)
                break  # Currently supports a single target branch per run

            train_loss = np.mean(train_loss)
            train_l1 = np.mean(train_l1)
            train_mse = np.mean(train_mse)

            result_dict['train_loss'].append(train_loss)
            result_dict['train_l1'].append(train_l1)
            result_dict['train_mse'].append(train_mse)

            print('\rTrain Epoch: {}, train_loss: {:.4f}, train_l1: {:.4f}, train_mse: {:.4f}'.format(
                epoch, train_loss, train_l1, train_mse))

            valid_loss, valid_l1, valid_mse, valid_corr, valid_ssim = [], [], [], [], []
            for index, loader in enumerate(self.branch_loaders):
                valid_loss_b, valid_l1_b, valid_mse_b, valid_corr_b, valid_ssim_b = \
                    self.valid_loop(loader[-1], index, use_mask=True, epoch=epoch)
                valid_loss.append(valid_loss_b)
                valid_l1.append(valid_l1_b)
                valid_mse.append(valid_mse_b)
                valid_corr.append(valid_corr_b)
                valid_ssim.append(valid_ssim_b)
                break  # Currently supports a single target branch per run

            valid_loss = np.mean(valid_loss)
            valid_l1 = np.mean(valid_l1)
            valid_mse = np.mean(valid_mse)
            valid_corr = np.mean(valid_corr)
            valid_ssim = np.mean(valid_ssim)

            # Combined validation metric: equal weighting of Pearson correlation and SSIM
            valid = 0.5 * valid_corr + 0.5 * valid_ssim

            result_dict['valid_loss'].append(valid_loss)
            result_dict['valid_l1'].append(valid_l1)
            result_dict['valid_mse'].append(valid_mse)
            result_dict['valid'].append(valid)

            print('\rValid Epoch: {}, valid_loss: {:.4f}, valid_l1: {:.4f}, valid_mse: {:.4f}'.format(
                epoch, valid_loss, valid_l1, valid_mse))
            print(f'Combined metric (0.5*corr + 0.5*ssim): {valid:.4f}\n')

            # Save a checkpoint when the combined metric improves, or every 10 epochs as a fallback
            if self.lowest_loss < valid or epoch % 10 == 0:
                model_params_dict = {
                    f'model_param_{idx}': model_param.state_dict()
                    for idx, model_param in enumerate(self.branch_models_g)
                }
                print('--------------------Saving best model--------------------')
                check_point = f"checkpoint_{epoch}.pt"
                check_point_d = f"checkpoint_d_{epoch}.pt"
                torch.save(model_params_dict, os.path.join(self.results_dir, check_point))
                torch.save(self.model_d.state_dict(), os.path.join(self.results_dir, check_point_d))
                self.lowest_loss = valid
                self.counter = 0
            else:
                self.counter += 1
                print('Loss is not decreased in last %d epochs' % self.counter)

            if (self.counter > patience) and (epoch >= minimum_epochs):
                break

            total_time = time.time() - start_time
            print('Time to process epoch({}): {:.4f} minutes\n'.format(epoch, total_time / 60))
            pd.DataFrame.from_dict(result_dict).to_csv(
                os.path.join(self.results_dir, 'training_stats.csv'), index=False)

        return result_dict

    def train_loop(self, data_loader):
        """
        Training loop for a single epoch (used by the base class only).

        Args:
            data_loader (DataLoader): Data loader for the training split.

        Returns:
            tuple[float, float, float]: Average total, L1, and MSE loss over all batches.
        """
        total_error = 0
        total_error_l1 = 0
        total_error_mse = 0

        self.model_g.train()
        batch_count = len(data_loader)
        for batch_idx, (input_batch, output_batch, _, _) in enumerate(data_loader):
            self.model_g.zero_grad()

            input_batch = input_batch.to(self.device)
            output_batch = output_batch.to(self.device)

            generated_output_batch = self.model_g(input_batch)

            error_l1 = self.loss_l1(output_batch, generated_output_batch)
            error_mse = self.loss_mse(output_batch, generated_output_batch)
            error = error_l1 + error_mse

            error.backward()
            self.optimizer.step()

            print('Training - [%d/%d]\tL1 Loss: %.06f \tMSE Loss: %.06f'
                  % (batch_idx, batch_count, error_l1.item(), error_mse.item()), end='\r')
            total_error += error.item()
            total_error_l1 += error_l1.item()
            total_error_mse += error_mse.item()

        return total_error / batch_count, total_error_l1 / batch_count, total_error_mse / batch_count

    def valid_loop(self, data_loader):
        """
        Validation loop for a single epoch (used by the base class only).

        Args:
            data_loader (DataLoader): Data loader for the validation split.

        Returns:
            tuple[float, float, float]: Average total, L1, and MSE loss over all batches.
        """
        total_error = 0
        total_error_l1 = 0
        total_error_mse = 0

        self.model_g.eval()
        batch_count = len(data_loader)
        with torch.no_grad():
            for batch_idx, (input_batch, output_batch, _, _) in enumerate(data_loader):
                input_batch = input_batch.to(self.device)
                output_batch = output_batch.to(self.device)

                generated_output_batch = self.model_g(input_batch)

                error_l1 = self.loss_l1(output_batch, generated_output_batch)
                error_mse = self.loss_mse(output_batch, generated_output_batch)
                error = error_l1 + error_mse

                print('Validation - [%d/%d]\tL1 Loss: %.06f \tMSE Loss: %.06f'
                      % (batch_idx, batch_count, error_l1.item(), error_mse.item()), end='\r')
                total_error += error.item()
                total_error_l1 += error_l1.item()
                total_error_mse += error_mse.item()

        return total_error / batch_count, total_error_l1 / batch_count, total_error_mse / batch_count

    def load_model(self, ckpt_path, model_g, index):
        """
        Loads generator weights from a checkpoint file.

        Args:
            ckpt_path (str): Path to the checkpoint .pt file.
            model_g (nn.Module): Generator instance to load weights into.
            index (int): Branch index used as the key inside the checkpoint dict.

        Returns:
            nn.Module: Generator with loaded weights, moved to self.device.
        """
        all_check_point = torch.load(ckpt_path, map_location=torch.device('cpu'))
        ckpt = all_check_point[f'model_param_{index}']
        # Strip 'module.' prefix that may be present from DataParallel-wrapped checkpoints
        ckpt_clean = {key.replace('module.', ''): val for key, val in ckpt.items()}
        model_g.load_state_dict(ckpt_clean, strict=True)
        model_g = model_g.to(device=self.device)
        return model_g

    def load_model_d(self, ckpt_path):
        """
        Loads discriminator weights from a checkpoint file.

        Args:
            ckpt_path (str): Path to the discriminator checkpoint .pt file.
        """
        ckpt = torch.load(ckpt_path)
        ckpt_clean = {key.replace('module.', ''): val for key, val in ckpt.items()}
        self.model_d.load_state_dict(ckpt_clean, strict=True)
        if torch.cuda.is_available():
            self.model_d = self.model_d.to(device=self.device)
        else:
            self.model_d = self.model_d.to(device=self.device).to(dtype=torch.bfloat16)

    def eval(self, data_csv_path, split_name='test', img_size=256, batch_size=64,
             num_workers=4, required_stains=[], checkpoint_name='checkpoint.pt'):
        """
        Runs inference on a held-out split and saves per-image evaluation metrics.

        For each stain in required_stains, builds the input channel list (full panel minus
        the target stain and all other potential output markers), loads the corresponding
        generator branch from the checkpoint, and runs eval_loop.

        Args:
            data_csv_path (str): Path to the CSV file with image paths and split labels.
            split_name (str): Name of the data split to evaluate. Defaults to 'test'.
            img_size (int): Spatial resolution for inference. Defaults to 256.
            batch_size (int): Batch size. Defaults to 64.
            num_workers (int): Number of data-loading worker processes. Defaults to 4.
            required_stains (list): Stain names to impute and evaluate.
            checkpoint_name (str): Filename of the generator checkpoint inside results_dir.
                                   Defaults to 'checkpoint.pt'.
        """
        data_loaders = []
        indexs = []
        model_gs = []

        # Other potential output markers that must be excluded from input at eval time
        other_pot_outputs = [m for m in self.pot_output_markers if m not in required_stains]

        for id, stain in enumerate(required_stains):
            self.set_seed(self.seed)
            input_marker = self.marker_panel.copy()
            input_marker.remove(stain)

            # Identify non-fixed input channels (candidates for masking)
            left_pot_marker = input_marker.copy()
            for marker in self.fixed_markers:
                left_pot_marker.remove(marker)

            # Exclude other potential output markers from input (not available at test time)
            for marker in other_pot_outputs:
                if marker in input_marker:
                    input_marker.remove(marker)

            stain_index = [input_marker.index(element)
                           for element in input_marker if element in left_pot_marker]
            self.stain_indexes.append(stain_index)

            dataset = MxIFReader(data_csv_path=data_csv_path, split_name=split_name,
                                 marker_panel=self.marker_panel,
                                 input_markers=input_marker, output_markers=[stain],
                                 training=False, img_size=img_size, percent=1)
            data_loader = MxIFReader.get_data_loader(dataset, batch_size=batch_size,
                                                     training=False, num_workers=num_workers)

            model_g = self.init_model(is_train=False, input_marker=input_marker,
                                      output_marker=[stain], had_d=True)
            # model_g = self.load_model(
            #     ckpt_path=os.path.join(self.results_dir, checkpoint_name),
            #     model_g=model_g, index=id
            # )
            data_loaders.append(data_loader)
            indexs.append(stain_index)
            model_gs.append(model_g)

        eval_dir_name = '%s_%d_%d' % (split_name, img_size, img_size)
        self.eval_loop(data_loaders, eval_dir_name, model_gs, indexs, required_stains)

    def mask_input_batchs(self, input_batch, stain_index, numbers):
        """
        Zeros out specific input channels to simulate missing stains at inference time.

        Args:
            input_batch (torch.Tensor): Input image tensor of shape (B, C, H, W).
            stain_index (list): Channel indices that are candidates for masking.
            numbers (list): Specific channel indices to zero out.

        Returns:
            torch.Tensor: Masked copy of the input batch.
        """
        if len(numbers) == 0:
            return input_batch.clone()

        masked_input_batch = input_batch.clone()
        for num in numbers:
            masked_input_batch[:, num, :, :] = 0
        return masked_input_batch

    def eval_loop(self, data_loaders, eval_dir_name, model_gs, indexs, stain_names):
        """
        Inference and metric computation loop for the evaluation split.

        For each image, saves a .npy file containing real and generated stain channels
        concatenated along the channel axis, and computes pixel-level metrics.

        Args:
            data_loaders (list): Data loaders, one per target stain.
            eval_dir_name (str): Sub-directory name under results_dir for saving outputs.
            model_gs (list): Generator models, one per stain.
            indexs (list): Maskable channel index lists, one per stain.
            stain_names (list): Names of the target stains, used for output directory naming.
        """
        stats_dict = {
            "Image_Name": [], "Stain": [], "MAE": [], "MSE": [],
            "SSIM": [], "PSNR": [], "RMSE": [], "Corr": [], "p-value": []
        }

        for id, data_loader in enumerate(data_loaders):
            model_g = model_gs[id].to(torch.float32).to(self.device)
            stain_name = stain_names[id]
            batch_count = len(data_loader)
            model_g.eval()

            stain_index = self.stain_indexes[id]
            os.makedirs(os.path.join(self.results_dir, eval_dir_name, stain_name), exist_ok=True)

            with torch.no_grad():
                for batch_idx, (input_batch, output_batch, image_name_batch, img_dims) in enumerate(data_loader):
                    # Pass an empty list to use all available input channels (no masking at eval)
                    input_batch = self.mask_input_batchs(input_batch, stain_index, [])
                    input_batch = input_batch.to(self.device).to(torch.float32)
                    output_batch = output_batch.to(self.device).to(torch.float32)

                    generated_batch = model_g(input_batch)

                    for i, image_name in enumerate(image_name_batch):
                        # Retrieve original spatial dimensions to remove any padding
                        img_dim = [img_dims[0][i].item(), img_dims[1][i].item()]
                        image_name = os.path.basename(image_name)
                        image_name, ext = os.path.splitext(image_name)

                        print('%d/%d - (%d) %s' % (batch_idx, batch_count, i, image_name))
                        input = input_batch[i, :, :, :].detach().cpu().numpy()
                        real = output_batch[i, :, :, :].detach().cpu().numpy()
                        generated = generated_batch[i, :, :, :].detach().cpu().numpy()

                        # Crop to original image size (removes padding introduced by the data loader)
                        input = input[:, :img_dim[0], :img_dim[1]]
                        real = real[:, :img_dim[0], :img_dim[1]]
                        generated = generated[:, :img_dim[0], :img_dim[1]]

                        # Save real and generated channels together for downstream analysis
                        output = np.concatenate([real, generated], axis=0)
                        np.save(os.path.join(self.results_dir, eval_dir_name, stain_name,
                                             image_name + '.npy'), output)

                        # Scale from [0, 1] to [0, 255] for metric computation
                        real = real * 255.0
                        generated = generated * 255.0

                        stats = self.pixel_metrics(real, generated, max_val=255, baseline=False)

                        stats_dict["Image_Name"].append(image_name)
                        stats_dict["Stain"].append(stain_name)
                        for key in stats.keys():
                            stats_dict[key].append(stats[key])

        pd.DataFrame.from_dict(stats_dict).to_csv(
            os.path.join(self.results_dir, '%s_stats.csv' % eval_dir_name), index=False)

    @staticmethod
    def pixel_metrics(real, generated, max_val=255, baseline=False):
        """
        Computes pixel-level evaluation metrics between the ground truth and generated images.

        Args:
            real (np.ndarray): Ground truth image array.
            generated (np.ndarray): Model-generated image array.
            max_val (float): Maximum pixel value, used for PSNR computation. Defaults to 255.
            baseline (bool): If True, skips Pearson correlation (used for baseline comparisons).

        Returns:
            dict: Dictionary with MAE, MSE, RMSE, PSNR, SSIM, and optionally Corr/p-value.
        """
        real = np.squeeze(real)
        generated = np.squeeze(generated)
        stats = {}
        stats["MAE"] = np.mean(np.abs(real - generated))
        stats["MSE"] = np.mean((real - generated) ** 2)
        stats["RMSE"] = np.sqrt(stats["MSE"])
        stats["PSNR"] = 20 * np.log10(max_val) - 10.0 * np.log10(stats["MSE"])
        stats["SSIM"] = ssim(real, generated, data_range=max_val)
        if not baseline:
            corr, p_value = st.pearsonr(real.flatten(), generated.flatten())
            stats['Corr'] = corr
            stats['p-value'] = p_value
        return stats

    def attributions(self, data_csv_path, split_name='test', img_size=256, batch_size=32,
                     num_workers=4):
        """
        Computes Integrated Gradients attributions for the test data using Captum.

        Args:
            data_csv_path (str): Path to the data CSV file.
            split_name (str): Name of the data split to attribute. Defaults to 'test'.
            img_size (int): Spatial resolution. Defaults to 256.
            batch_size (int): Batch size. Defaults to 32.
            num_workers (int): Number of data-loading worker processes. Defaults to 4.
        """
        self.set_seed(self.seed)
        dataset = MxIFReader(data_csv_path=data_csv_path, split_name=split_name,
                             marker_panel=self.marker_panel,
                             input_markers=self.input_markers,
                             output_markers=self.output_markers,
                             training=False, img_size=img_size)
        data_loader = MxIFReader.get_data_loader(dataset, batch_size=batch_size,
                                                 training=False, num_workers=num_workers)
        self.init_model()
        self.load_model(ckpt_path=os.path.join(self.results_dir, 'checkpoint.pt'))
        attr_dir_name = 'attributions_%s_%d_%d' % (split_name, img_size, img_size)
        self.attributions_loop(data_loader, attr_dir_name)

    def attributions_loop(self, data_loader, attr_dir_name):
        """
        Attribution computation loop using Integrated Gradients.

        Saves per-image attribution arrays (absolute, positive, negative) as .npy files
        and aggregates per-channel attribution scalars across the dataset into CSV files.

        Args:
            data_loader (DataLoader): Data loader for the split to attribute.
            attr_dir_name (str): Directory name for saving attribution outputs.
        """
        os.makedirs(os.path.join(self.results_dir, attr_dir_name), exist_ok=True)
        image_path_list = []
        attr_array = None
        attr_array_pos = None
        attr_array_neg = None
        batch_count = len(data_loader)
        ig = IntegratedGradients(self.interpretable_model)
        self.model_g.eval()

        for batch_idx, (input_batch, output_batch, image_name_batch, img_dims) in enumerate(data_loader):
            input_batch = input_batch.to(self.device)
            input_batch.requires_grad_()
            attr, _ = ig.attribute(input_batch,
                                   baselines=torch.zeros_like(input_batch, device=self.device),
                                   target=0, return_convergence_delta=True)

            for i, image_name in enumerate(image_name_batch):
                image_path_list.append(image_name)
                image_name = os.path.basename(image_name)
                image_name, ext = os.path.splitext(image_name)
                print('%d/%d - %s' % (batch_idx, batch_count, image_name))

                img_dim = [img_dims[0][i].item(), img_dims[1][i].item()]
                attr = attr.detach().cpu().numpy()
                attr_ = attr[i, :, :img_dim[0], :img_dim[1]]
                np.save(os.path.join(self.results_dir, attr_dir_name, image_name + '_attr.npy'), attr_)

                pos_attr_ = np.maximum(attr_, 0)
                neg_attr_ = np.maximum((-1) * attr_, 0)

                # Sum over spatial dimensions to obtain a scalar attribution per channel
                attr_ = np.expand_dims(np.sum(np.sum(np.abs(attr_), axis=-1), axis=-1), axis=0)
                pos_attr_ = np.expand_dims(np.sum(np.sum(pos_attr_, axis=-1), axis=-1), axis=0)
                neg_attr_ = np.expand_dims(np.sum(np.sum(neg_attr_, axis=-1), axis=-1), axis=0)

                if attr_array is None:
                    attr_array = attr_
                    attr_array_pos = pos_attr_
                    attr_array_neg = neg_attr_
                else:
                    attr_array = np.concatenate((attr_array, attr_), axis=0)
                    attr_array_pos = np.concatenate((attr_array_pos, pos_attr_), axis=0)
                    attr_array_neg = np.concatenate((attr_array_neg, neg_attr_), axis=0)

        marker_names = data_loader.dataset.input_markers
        df = pd.DataFrame(attr_array, columns=marker_names)
        df_pos = pd.DataFrame(attr_array_pos, columns=marker_names)
        df_neg = pd.DataFrame(attr_array_neg, columns=marker_names)
        df['image_path'] = image_path_list
        df_pos['image_path'] = image_path_list
        df_neg['image_path'] = image_path_list
        df_pos.to_csv(os.path.join(self.results_dir, '%s_attributions_pos.csv' % attr_dir_name), index=False)
        df_neg.to_csv(os.path.join(self.results_dir, '%s_attributions_neg.csv' % attr_dir_name), index=False)
        df.to_csv(os.path.join(self.results_dir, '%s_attributions_abs.csv' % attr_dir_name), index=False)

    def interpretable_model(self, batch):
        """
        Wrapper around the generator that returns a spatially-pooled scalar output.
        Required by Captum's Integrated Gradients, which expects a scalar target per sample.
        """
        pred = self.model_g(batch)
        pred = nn.AdaptiveAvgPool2d((1, 1))(pred)
        return pred


def read_json_from_txt(file_path):
    """
    Reads a JSON-formatted text file and returns the marker names as an ordered list.

    The file is expected to contain a JSON object whose values are strings of the form
    '<marker_name> <optional_extra_info>'. Only the marker name (first token) is retained.

    Args:
        file_path (str): Path to the .txt file containing the JSON data.

    Returns:
        list[str]: Ordered list of marker names.
    """
    with open(file_path, "r") as file:
        data = file.read()

    original_dict = json.loads(data)
    values_list = []
    for value in original_dict.values():
        value = value.split(' ')[0]
        values_list.append(value)

    return values_list
