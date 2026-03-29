"""
trainer_AdSIMIMO.py — Base Trainer class for AdSI-MIMO.

Provides the training loop orchestration, checkpoint management, and evaluation
pipeline that are shared across all model variants. The concrete model, optimizer,
and loss function are supplied by subclasses (e.g., TrainerMMAE in train_AdSIMIMO.py).
"""

import json
import os
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats as st
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import platform

from dataloader_ssim import MxIFReader


class Trainer:
    def __init__(self, marker_panel, fixed_stain, results_dir, lr=0.002, seed=1):
        """
        Base trainer for MxIF stain imputation models.

        Args:
            marker_panel (list): Ordered list of marker names matching the channel order
                                 in the .npy image files.
            fixed_stain (list): Markers that are always available as input and never
                                imputed (e.g., ['dapi', 'autofluorescence']).
            results_dir (str): Directory where checkpoints and result files are saved.
            lr (float): Initial learning rate. Defaults to 0.002.
            seed (int): Global random seed for reproducibility. Defaults to 1.
        """
        self.marker_panel = marker_panel
        self.fixed_stain = fixed_stain
        self.results_dir = results_dir
        self.lr = lr
        self.seed = seed

        self.counter = 0
        self.lowest_loss = np.Inf
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None
        self.loss_balancer = None
        self.tasks_loss_fn = None
        self.stain_indexes = []
        self.input_domains = []
        self.output_domains = []
        self.img_size = None

        os.makedirs(self.results_dir, exist_ok=True)

    def set_seed(self, seed):
        """
        Sets the random seed for Python, NumPy, and PyTorch for reproducibility.

        Args:
            seed (int): Seed value.
        """
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        if platform.system() == 'Darwin':
            torch.use_deterministic_algorithms(True)

    def init_data_loader(self, data_csv_path, percent=100, img_size=256,
                         batch_size=64, num_workers=4, input_marker=None):
        """
        Builds training and validation DataLoaders.

        Args:
            data_csv_path (str): Path to the CSV file with image paths and split labels.
            percent (int): Percentage of training samples to use. Defaults to 100.
            img_size (int): Spatial resolution passed to MxIFReader. Defaults to 256.
            batch_size (int): Batch size for both loaders. Defaults to 64.
            num_workers (int): Worker processes for data loading. Defaults to 4.
            input_marker (list): Marker names to use as model input channels.

        Returns:
            list: [train_dataset, train_loader, valid_dataset, valid_loader]
        """
        if input_marker is None:
            input_marker = []

        train_dataset = MxIFReader(data_csv_path=data_csv_path, split_name='train',
                                   marker_panel=self.marker_panel,
                                   input_markers=input_marker,
                                   training=True, img_size=img_size, percent=percent)
        train_loader = MxIFReader.get_data_loader(train_dataset, batch_size=batch_size,
                                                  training=True, num_workers=num_workers)

        valid_dataset = MxIFReader(data_csv_path=data_csv_path, split_name='valid',
                                   marker_panel=self.marker_panel,
                                   input_markers=input_marker,
                                   training=False, img_size=img_size)
        valid_loader = MxIFReader.get_data_loader(valid_dataset, batch_size=batch_size,
                                                  training=False, num_workers=num_workers)

        return [train_dataset, train_loader, valid_dataset, valid_loader]

    def train(self, data_csv_path, percent=100, img_size=256, batch_size=64,
              num_workers=4, max_epochs=200, minimum_epochs=50, patience=25,
              load_model_ckpt=False, checkpoint_name=None):
        """
        Full training pipeline: data loading, model/optimizer initialization,
        epoch loop with early stopping, and periodic checkpointing.

        Args:
            data_csv_path (str): Path to the CSV file with image paths and split labels.
            percent (int): Percentage of training data to use per epoch. Defaults to 100.
            img_size (int): Spatial resolution for training patches. Defaults to 256.
            batch_size (int): Batch size. Defaults to 64.
            num_workers (int): Worker processes for data loading. Defaults to 4.
            max_epochs (int): Maximum number of training epochs. Defaults to 200.
            minimum_epochs (int): Minimum epochs before early stopping is allowed. Defaults to 50.
            patience (int): Epochs without improvement before stopping. Defaults to 25.
            load_model_ckpt (bool): If True, resumes from an existing checkpoint. Defaults to False.
            checkpoint_name (str): Checkpoint filename to resume from (required if
                                   load_model_ckpt=True).

        Returns:
            dict: Training history with per-epoch train/valid losses and per-domain SSIM/Corr.
        """
        self.counter = 0
        self.lowest_loss = np.Inf
        self.set_seed(seed=self.seed)
        self.img_size = img_size

        # Full marker panel is used as input; fixed stains are excluded from output
        input_marker = self.marker_panel.copy()
        output_marker = [m for m in self.marker_panel if m not in self.fixed_stain]

        self.input_domains = input_marker.copy()
        self.output_domains = output_marker.copy()

        data_loaders = self.init_data_loader(data_csv_path, percent=percent,
                                             img_size=img_size, batch_size=batch_size,
                                             num_workers=num_workers,
                                             input_marker=input_marker)

        self.model, DOMAIN_CONF = self.init_model(self.input_domains, self.output_domains)
        self.loss_balancer, self.tasks_loss_fn = self.init_loss_function(
            self.output_domains, DOMAIN_CONF
        )
        self.optimizer = self.init_optimizer(
            model={'model': self.model, 'balancer': self.loss_balancer}
        )

        start_epoch = 0
        if load_model_ckpt:
            ckpt_path = os.path.join(self.results_dir, checkpoint_name)
            start_epoch = self.load_mae_model(ckpt_path) + 1
            print(f"Resuming training from epoch {start_epoch}.")

        # Initialize result dictionary with per-domain metric columns
        result_dict = {'train_loss': [], 'valid_loss': []}
        for domain in self.output_domains:
            result_dict[f'ssim_{domain}'] = []
            result_dict[f'corr_{domain}'] = []

        for epoch in range(start_epoch, max_epochs):
            start_time = time.time()

            train_loss = self.train_loop(data_loaders[1], epoch,
                                         self.input_domains, self.output_domains)
            result_dict['train_loss'].append(train_loss)
            print(f'\rTrain  Epoch {epoch:04d}  loss={train_loss:.4f}')

            valid_loss, corr, ssim_ = self.valid_loop(data_loaders[3], epoch,
                                                       self.input_domains, self.output_domains)
            result_dict['valid_loss'].append(valid_loss)
            for i, domain in enumerate(self.output_domains):
                result_dict[f'ssim_{domain}'].append(ssim_[i])
                result_dict[f'corr_{domain}'].append(corr[i])
            print(f'\rValid  Epoch {epoch:04d}  loss={valid_loss:.4f}')

            # Save on improvement or as periodic fallback every 15 epochs
            if self.lowest_loss > valid_loss or epoch % 15 == 0:
                ckpt_path = os.path.join(self.results_dir, f'checkpoint_{epoch}.pt')
                self.save_mae_model(epoch, self.model, self.optimizer,
                                    self.loss_balancer, ckpt_path)
                print('--------------------Saving best model--------------------')
                self.lowest_loss = valid_loss
                self.counter = 0
            else:
                self.counter += 1
                print(f'No improvement for {self.counter} epoch(s).')

            if self.counter > patience and epoch >= minimum_epochs:
                print('Early stopping triggered.')
                break

            elapsed = (time.time() - start_time) / 60
            print(f'Epoch {epoch} completed in {elapsed:.2f} min\n')
            pd.DataFrame.from_dict(result_dict).to_csv(
                os.path.join(self.results_dir, 'training_stats.csv'), index=False
            )

        return result_dict

    def save_mae_model(self, epoch, model, optimizer, loss_balancer, save_path):
        """
        Saves model weights, optimizer state, loss balancer, and epoch index.

        Args:
            epoch (int): Current training epoch.
            model (nn.Module): The main network (MultiMAE backbone + adapters).
            optimizer (Optimizer): The optimizer instance.
            loss_balancer (nn.Module): The uncertainty weighting module.
            save_path (str): Full file path for the checkpoint.
        """
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss_balancer': loss_balancer.state_dict(),
        }, save_path)

    def load_mae_model(self, ckpt_path):
        """
        Loads model weights, optimizer state, and loss balancer from a checkpoint.

        Optimizer tensors are moved to self.device after loading to avoid
        device mismatches when resuming on a different GPU.

        Args:
            ckpt_path (str): Path to the checkpoint .pt file.

        Returns:
            int: The epoch at which the checkpoint was saved.
        """
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Model weights loaded from {ckpt_path}.")

        if 'optimizer' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # Move all optimizer state tensors to the target device after loading from CPU
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            print(f"Optimizer state loaded and moved to {self.device}.")
        else:
            print("Warning: optimizer state not found or optimizer not initialized.")

        if 'loss_balancer' in checkpoint and self.loss_balancer is not None:
            self.loss_balancer.load_state_dict(checkpoint['loss_balancer'])
            print(f"Loss balancer state loaded from {ckpt_path}.")
        else:
            print("Warning: loss_balancer state not found or not initialized.")

        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resuming from epoch {start_epoch}.")
        return start_epoch

    def min_max_normalize(self, tensor):
        """
        Applies per-sample, per-channel min-max normalization to a (B, C, H, W) tensor.

        Args:
            tensor (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Normalized tensor with values in [0, 1].
        """
        min_val = tensor.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        max_val = tensor.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        return (tensor - min_val) / (max_val - min_val + 1e-8)

    def eval(self, data_csv_path, split_name='test', img_size=256, batch_size=64,
             num_workers=4, checkpoint_name='checkpoint.pt',mask_biomarker = []):
        """
        Runs inference on a held-out split and saves per-image evaluation metrics.

        All non-fixed output markers are imputed simultaneously: their input channels
        are zeroed out so the model must reconstruct them from the fixed input channels.

        Args:
            data_csv_path (str): Path to the CSV file with image paths and split labels.
            split_name (str): Split to evaluate ('test' by default).
            img_size (int): Spatial resolution for inference. Defaults to 256.
            batch_size (int): Batch size. Defaults to 64.
            num_workers (int): Worker processes for data loading. Defaults to 4.
            checkpoint_name (str): Filename of the checkpoint inside results_dir.
                                   Defaults to 'checkpoint.pt'.
        """
        input_markers = self.marker_panel.copy()
        output_markers = [m for m in self.marker_panel if m not in self.fixed_stain]

        self.input_domains = input_markers.copy()
        self.output_domains = output_markers.copy()
        self.img_size = img_size

        self.set_seed(self.seed)

        dataset = MxIFReader(data_csv_path=data_csv_path, split_name=split_name,
                             marker_panel=self.marker_panel,
                             input_markers=input_markers,
                             training=False, img_size=img_size, percent=1)
        data_loader = MxIFReader.get_data_loader(dataset, batch_size=batch_size,
                                                 training=False, num_workers=num_workers)

        self.model, DOMAIN_CONF = self.init_model_eval(self.input_domains, self.output_domains)
        self.loss_balancer, self.tasks_loss_fn = self.init_loss_function(
            self.output_domains, DOMAIN_CONF
        )
        # optimizer is not needed for inference; set to None so load_mae_model skips it
        self.optimizer = None

        ckpt_path = os.path.join(self.results_dir, checkpoint_name)
        self.load_mae_model(ckpt_path)

        eval_dir_name = f'{split_name}_{img_size}_{img_size}'
        self.eval_loop(data_loader, eval_dir_name, self.model, mask_biomarker)

    def eval_loop(self, data_loader, eval_dir_name, model_mae, mask_biomarker):
        """
        Inference loop that imputes the specified stains and saves per-image metrics.

        Only the stains listed in mask_biomarker are zeroed out and reconstructed;
        all other output-domain stains remain visible to the encoder as context.
        Results (.npy arrays of [real, generated]) and a per-domain CSV of pixel-level
        metrics are saved under results_dir/eval_dir_name/.

        Args:
            data_loader (DataLoader): DataLoader for the evaluation split.
            eval_dir_name (str): Sub-directory name under results_dir for saving outputs.
            model_mae (nn.Module): Trained MultiMAE model in evaluation mode.
            mask_biomarker (list[str]): Output-domain markers to impute (zeroed in encoder
                input); any output domain not listed here acts as context.
        """
        real_output_index = [
            self.output_domains.index(d)
            for d in self.output_domains
            if d in mask_biomarker
        ]
        decode_domains = [self.output_domains[i] for i in real_output_index]

        # Initialize per-domain statistics dictionary (only for imputed domains)
        stats_dict = {
            domain: {
                'Image_Name': [], 'Stain': [], 'MAE': [], 'MSE': [],
                'SSIM': [], 'PSNR': [], 'RMSE': [], 'Corr': [], 'p-value': []
            }
            for domain in decode_domains
        }

        model_mae = model_mae.to(torch.float32).to(self.device)
        batch_count = len(data_loader)
        model_mae.eval()

        with torch.no_grad():
            for batch_idx, (input_batch, image_name_batch, img_dims) in tqdm(
                    enumerate(data_loader), total=batch_count, desc='Evaluating'):

                input_batch = input_batch.to(self.device)

                # Build per-domain dict: each entry is one channel slice (B, 1, H, W)
                mae_batch = {
                    self.input_domains[i]: input_batch[:, i:i + 1, :, :]
                    for i in range(len(self.input_domains))
                }

                num_context_odomains = len(self.output_domains) - len(real_output_index)
                num_encoded_tokens = int(num_context_odomains * np.square(self.img_size / 16))

                # Zero out only the target stains; context output stains remain visible
                inputmodel_batch = mae_batch.copy()
                for biomarker in mask_biomarker:
                    if biomarker in inputmodel_batch:
                        inputmodel_batch[biomarker] = torch.zeros_like(mae_batch[biomarker])

                outputs, task_masks = model_mae(x=inputmodel_batch,
                                                num_encoded_tokens=num_encoded_tokens,
                                                real_output_index=real_output_index)

                # Collect generated and real images for imputed domains only
                fake_image = np.concatenate(
                    [outputs[d].to(torch.float32).detach().cpu().numpy()
                     for d in decode_domains], axis=1
                )
                real_image = np.concatenate(
                    [mae_batch[d].to(torch.float32).detach().cpu().numpy()
                     for d in decode_domains], axis=1
                )

                for k, domain in enumerate(decode_domains):
                    domain_dir = os.path.join(self.results_dir, eval_dir_name, domain)
                    os.makedirs(domain_dir, exist_ok=True)

                    for j, img_path in enumerate(image_name_batch):
                        img_dim = [img_dims[0][j].item(), img_dims[1][j].item()]
                        image_name = os.path.splitext(os.path.basename(img_path))[0]

                        print(f'{batch_idx}/{batch_count} - ({j}) {image_name}')

                        real = real_image[j, k:k + 1, :img_dim[0], :img_dim[1]]
                        generated = fake_image[j, k:k + 1, :img_dim[0], :img_dim[1]]

                        # Save concatenated [real, generated] for downstream analysis
                        np.save(os.path.join(domain_dir, image_name + '.npy'),
                                np.concatenate([real, generated], axis=0))

                        # Scale to [0, 255] for metric computation
                        stats = self.pixel_metrics(real * 255.0, generated * 255.0,
                                                   max_val=255, baseline=False)

                        stats_dict[domain]['Image_Name'].append(image_name)
                        stats_dict[domain]['Stain'].append(domain)
                        for key, val in stats.items():
                            stats_dict[domain][key].append(val)

        # Write per-domain CSV files (only for imputed domains)
        for domain in decode_domains:
            csv_path = os.path.join(self.results_dir, eval_dir_name,
                                    f'{domain}_stats.csv')
            pd.DataFrame(stats_dict[domain]).to_csv(csv_path, index=False)

    @staticmethod
    def pixel_metrics(real, generated, max_val=255, baseline=False):
        """
        Computes pixel-level evaluation metrics between a ground-truth and generated image.

        Args:
            real (np.ndarray): Ground-truth image array.
            generated (np.ndarray): Model-generated image array.
            max_val (float): Maximum pixel value used for PSNR. Defaults to 255.
            baseline (bool): If True, skips Pearson correlation (for baseline comparisons).

        Returns:
            dict: MAE, MSE, RMSE, PSNR, SSIM, and (unless baseline=True) Corr and p-value.
        """
        real = np.squeeze(real)
        generated = np.squeeze(generated)
        stats = {
            'MAE':  np.mean(np.abs(real - generated)),
            'MSE':  np.mean((real - generated) ** 2),
        }
        stats['RMSE'] = np.sqrt(stats['MSE'])
        stats['PSNR'] = 20 * np.log10(max_val) - 10.0 * np.log10(stats['MSE'] + 1e-8)
        stats['SSIM'] = ssim(real, generated, data_range=max_val)
        if not baseline:
            corr, p_val = st.pearsonr(real.flatten(), generated.flatten())
            stats['Corr'] = corr
            stats['p-value'] = p_val
        return stats


def read_json_from_txt(file_path):
    """
    Reads a JSON-formatted text file and returns an ordered list of marker names.

    The expected format is a JSON object whose values are strings of the form
    'marker_name ...' (only the first space-separated token is used).

    Args:
        file_path (str): Path to the JSON-formatted .txt file.

    Returns:
        list[str]: Ordered list of marker names.
    """
    with open(file_path, 'r') as f:
        original_dict = json.loads(f.read())
    return [v.split(' ')[0] for v in original_dict.values()]
