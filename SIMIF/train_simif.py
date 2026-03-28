import platform
import torch
import torch.nn as nn
import torch.optim as optim

from networks_base import initialize_weights, Generator, Discriminator, weights_init
from trainer_simif import read_json_from_txt, Trainer
import numpy as np
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim
import random


class TrainerCGAN(Trainer):
    def __init__(self,
                 marker_panel,
                 fixed_markers,
                 potential_output_markers,
                 results_dir,
                 target_marker,
                 lr=0.002,
                 seed=1):
        """
        Trainer subclass that uses a Wasserstein GAN with Gradient Penalty (WGAN-GP)
        for stain imputation.

        Args:
            marker_panel (list): Marker names in channel order for the MxIF images.
            fixed_markers (list): Markers always present as input (e.g., DAPI, autofluorescence).
            potential_output_markers (list): Markers that can serve as imputation targets.
            results_dir (str): Directory for saving checkpoints and training statistics.
            target_marker (list): Specific marker(s) to impute in this training run.
            lr (float): Learning rate for the Adam optimizer. Defaults to 0.002.
            seed (int): Random seed for reproducibility. Defaults to 1.
        """
        super().__init__(marker_panel,
                         fixed_markers,
                         potential_output_markers,
                         results_dir,
                         target_marker,
                         lr)

        self.model_d = None
        # Mixed-precision scalers are only used on CUDA; set to None on macOS (MPS/CPU)
        if platform.system() == 'Darwin':
            self.g_scaler = None
            self.d_scaler = None
        else:
            self.g_scaler = torch.cuda.amp.GradScaler()
            self.d_scaler = torch.cuda.amp.GradScaler()
        self.loss_bce = None
        self.mask = True
        self.optimizer_d = None

    def init_model(self, is_train=False, input_marker=[], output_marker=[], had_d=False):
        """
        Initializes the generator and, optionally, a shared discriminator.

        When had_d=False (first branch), both a generator and discriminator are created.
        When had_d=True (subsequent branches), only a generator is created because the
        discriminator is shared across all branches.

        Args:
            is_train (bool): If True, moves models to the compute device. Defaults to False.
            input_marker (list): Input channel names.
            output_marker (list): Output channel names.
            had_d (bool): Whether the shared discriminator already exists. Defaults to False.

        Returns:
            nn.Module or tuple[nn.Module, nn.Module]: Generator alone, or (generator, discriminator).
        """
        model_g = Generator(in_channels=len(input_marker),
                            out_channels=len(output_marker), init_features=128)
        model_g = model_g.apply(weights_init)

        if had_d:
            # Discriminator already exists; only return the new generator
            if is_train:
                model_g = model_g.to(device=self.device)
            return model_g
        else:
            model_d = Discriminator(real_channels=len(input_marker),
                                    gen_channels=len(output_marker))
            model_d = model_d.apply(weights_init)
            if is_train:
                model_g = model_g.to(device=self.device)
                model_d = model_d.to(device=self.device)
            return model_g, model_d

    def init_optimizer(self, model_g, model_d, has_o):
        """
        Initializes Adam optimizers for the generator and, on first call, the discriminator.

        The discriminator uses a learning rate of 2× the generator's rate, which is a common
        heuristic to keep the discriminator slightly ahead during adversarial training.

        Args:
            model_g (nn.Module): Generator model.
            model_d (nn.Module): Discriminator model.
            has_o (bool): Whether the discriminator optimizer already exists.

        Returns:
            optim.Adam or tuple[optim.Adam, optim.Adam]: Generator optimizer, or both optimizers.
        """
        optimizer = optim.Adam(model_g.parameters(), lr=self.lr, betas=(0.5, 0.999))
        if has_o:
            return optimizer
        else:
            self.optimizer_d = optim.Adam(model_d.parameters(), lr=self.lr * 2, betas=(0.5, 0.999))
            return optimizer, self.optimizer_d

    def init_loss_function(self):
        """Initializes L1, MSE, and BCE loss functions."""
        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCEWithLogitsLoss()

    def get_weights(self, current_epoch, interval=25):
        """
        Returns a curriculum probability distribution over the number of input channels
        to mask (0, 1, 2, or 3). The distribution gradually shifts from favouring no masking
        toward heavier masking as training progresses, encouraging the model to generalise
        to incomplete stain panels.

        Args:
            current_epoch (int): Current training epoch.
            interval (int): Number of epochs between each distribution shift. Defaults to 25.

        Returns:
            list[float]: Normalised probability distribution over [0, 1, 2, 3] masked channels.
        """
        probabilities = [0.7, 0.2, 0.1, 0]
        num_increments = current_epoch // interval

        for _ in range(num_increments):
            if probabilities[-1] == 1.0:
                break
            for i in range(len(probabilities)):
                if probabilities[i] > 0:
                    delta = min(probabilities[i], 0.1)
                    probabilities[i] -= delta

                    if i + 1 < len(probabilities):
                        probabilities[i + 1] += delta / 2
                    if i + 2 < len(probabilities):
                        probabilities[i + 2] += delta / 2
                    elif i + 1 == len(probabilities) - 1:
                        probabilities[i + 1] += delta

                    if i + 1 < len(probabilities):
                        probabilities[i + 1] = min(probabilities[i + 1], 1.0)
                    if probabilities[-1] == 1.0:
                        for j in range(len(probabilities) - 1):
                            probabilities[j] = 0.0
                        break
                    break

        total = sum(probabilities)
        if total > 0:
            probabilities = [p / total for p in probabilities]

        return probabilities

    def mask_input_batch(self, input_batch, stain_index, current_epoch):
        """
        Randomly zeros out a subset of the non-fixed input channels as a curriculum data
        augmentation strategy, simulating partially missing stain panels at training time.

        The number of channels to mask is sampled from the epoch-dependent distribution
        returned by get_weights(). Early in training most batches are unmasked; later,
        heavier masking becomes increasingly likely so the model learns to handle incomplete inputs.

        Args:
            input_batch (torch.Tensor): Input image tensor of shape (B, C, H, W).
            stain_index (list): Indices of the non-fixed (maskable) channels.
            current_epoch (int): Current training epoch, used to retrieve the curriculum weights.

        Returns:
            torch.Tensor: Masked copy of the input batch.
        """
        weights = self.get_weights(current_epoch)
        num_channel_mask = random.choices([0, 1, 2, 3], weights=weights, k=1)[0]
        numbers = random.sample(stain_index, num_channel_mask)

        masked_input_batch = input_batch.clone()
        for num in numbers:
            masked_input_batch[:, num, :, :] = 0
        return masked_input_batch

    def train_loop(self, data_loader, index, epoch):
        """
        Single-epoch WGAN-GP training loop.

        Alternates between one discriminator update (with gradient penalty) and one
        generator update (adversarial loss + L1 pixel reconstruction loss).

        Args:
            data_loader (DataLoader): Training data loader for the current branch.
            index (int): Branch index, used to select the correct generator and optimizer.
            epoch (int): Current epoch, forwarded to the curriculum masking scheduler.

        Returns:
            tuple[float, float, float]: Average combined, L1, and MSE loss over all batches.
        """
        total_error = 0
        total_error_l1 = 0
        total_error_mse = 0

        model_g = self.branch_models_g[index]
        stain_index = self.stain_indexes[index]
        model_g.train()
        self.model_d.train()
        batch_count = len(data_loader)

        for batch_idx, (input_batch, output_batch, _, _) in enumerate(data_loader):
            input_batch = input_batch.to(self.device, dtype=torch.float32)
            output_batch = output_batch.to(self.device, dtype=torch.float32)
            input_batch = self.mask_input_batch(input_batch, stain_index, epoch)

            # ------- Discriminator update -------
            real_ab = torch.cat([input_batch, output_batch], dim=1)
            y_fake = model_g(input_batch)
            fake_ab = torch.cat([input_batch, y_fake.detach()], dim=1)

            self.optimizer_d.zero_grad()

            def compute_gradient_penalty(D, real_samples, fake_samples):
                """
                WGAN-GP gradient penalty: penalises the discriminator when the norm of
                its gradients w.r.t. interpolated samples deviates from 1, enforcing the
                1-Lipschitz constraint required by the Wasserstein distance formulation.
                """
                alpha = torch.rand(real_samples.size(0), 1, 1, 1,
                                   dtype=torch.float32).to(real_samples.device)
                interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
                d_interpolates = D(interpolates)
                fake = torch.ones(d_interpolates.size(), dtype=torch.float32).to(real_samples.device)
                gradients = torch.autograd.grad(
                    outputs=d_interpolates, inputs=interpolates,
                    grad_outputs=fake, create_graph=True, retain_graph=True,
                    only_inputs=True
                )[0]
                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                return gradient_penalty

            D_real = self.model_d(real_ab)
            D_fake = self.model_d(fake_ab)
            # Wasserstein discriminator loss: maximise E[D(real)] - E[D(fake)]
            D_real_loss = -torch.mean(D_real)
            D_fake_loss = torch.mean(D_fake)
            D_loss = D_real_loss + D_fake_loss
            gradient_penalty = compute_gradient_penalty(self.model_d, real_ab, fake_ab)
            D_loss += 10 * gradient_penalty  # λ = 10 as recommended in the WGAN-GP paper

            D_loss.backward()
            self.optimizer_d.step()

            # ------- Generator update -------
            self.optimizers[index].zero_grad()

            y_fake = model_g(input_batch)
            fake_ab = torch.cat([input_batch, y_fake], dim=1)
            D_fake = self.model_d(fake_ab)
            # Generator minimises the negated discriminator score on fake samples
            G_fake_loss = -torch.mean(D_fake)
            L1 = self.loss_l1(y_fake, output_batch)
            L2 = self.loss_mse(y_fake, output_batch)
            # Combined generator loss: adversarial term + weighted L1 reconstruction (λ_L1 = 50)
            G_loss = G_fake_loss + L1 * 50

            G_loss.backward()
            self.optimizers[index].step()

            print('Training - [%d/%d] - D_Loss: %.06f - G_Loss: %.06f - L1_Loss: %.06f '
                  '- G_Fake_Loss: %.06f - D_Fake_Loss: %.06f' %
                  (batch_idx, batch_count, D_loss.item(), G_loss.item(),
                   L1.item(), G_fake_loss.item(), D_fake_loss.item()), end='\r')

            total_error += L1.item() + L2.item() + G_fake_loss.item()
            total_error_l1 += L1.item()
            total_error_mse += L2.item()

        return total_error / batch_count, total_error_l1 / batch_count, total_error_mse / batch_count

    def valid_loop(self, data_loader, index, use_mask=False, epoch=0):
        """
        Single-epoch validation loop with WGAN-GP losses and image-quality metrics.

        The input is masked using epoch+130 to reach the late-curriculum distribution,
        evaluating the model under heavy masking to measure worst-case imputation quality.

        Args:
            data_loader (DataLoader): Validation data loader for the current branch.
            index (int): Branch index.
            use_mask (bool): Unused; retained for API compatibility.
            epoch (int): Current epoch number.

        Returns:
            tuple[float, float, float, float, float]:
                Average combined loss, L1 loss, MSE loss, Pearson correlation, and SSIM.
        """
        total_error = 0
        total_error_l1 = 0
        total_error_mse = 0
        model_g = self.branch_models_g[index]

        model_g.eval()
        self.model_d.eval()

        batch_count = len(data_loader)
        real_images = []
        fake_images = []
        corr_list = []
        ssim_scores = []

        for batch_idx, (input_batch, output_batch, _, _) in enumerate(data_loader):
            input_batch = input_batch.to(self.device).to(torch.float32)
            output_batch = output_batch.to(self.device).to(torch.float32)
            # Offset epoch by 130 to sample from the late-training masking distribution,
            # testing the model under the heaviest masking conditions
            input_batch = self.mask_input_batch(input_batch, self.stain_indexes[index], epoch + 130)

            y_fake = model_g(input_batch).to(torch.float32)
            self.model_d = self.model_d.to(torch.float32)

            real_ab = torch.cat([input_batch, output_batch], dim=1)
            fake_ab = torch.cat([input_batch, y_fake], dim=1)

            with torch.no_grad():
                real_images.append(output_batch.to(torch.float32).cpu().numpy())
                fake_images.append(y_fake.to(torch.float32).cpu().numpy())

            with torch.no_grad():
                D_real = self.model_d(real_ab)
                D_fake = self.model_d(fake_ab.detach())
                D_real_loss = -torch.mean(D_real)
                D_fake_loss = torch.mean(D_fake)
                D_loss = D_fake_loss + D_real_loss

            with torch.no_grad():
                D_fake = self.model_d(fake_ab)
                G_fake_loss = -torch.mean(D_fake)
                L1 = self.loss_l1(y_fake, output_batch)
                L2 = self.loss_mse(y_fake, output_batch)
                G_loss = G_fake_loss + L1 * 10

            print('Validation - [%d/%d] - D_Loss: %.06f - G_Loss: %.06f - L1_Loss: %.06f '
                  '- G_Fake_Loss: %.06f - D_Fake_Loss: %.06f' %
                  (batch_idx, batch_count, D_loss.item(), G_loss.item(),
                   L1.item(), G_fake_loss.item(), D_fake_loss.item()), end='\r')

            total_error += L1.item() + L2.item() + G_fake_loss.item()
            total_error_l1 += L1.item()
            total_error_mse += L2.item()

        real_images = np.concatenate(real_images, axis=0)
        fake_images = np.concatenate(fake_images, axis=0)
        # Convert from (N, C, H, W) to (N, H, W, C) for scikit-image metric functions
        real_images = np.transpose(real_images, (0, 2, 3, 1))
        fake_images = np.transpose(fake_images, (0, 2, 3, 1))

        for i in range(real_images.shape[0]):
            # Skip constant-value images to avoid undefined Pearson correlation
            if (np.all(real_images[i].flatten() == real_images[i].flatten()[0]) or
                    np.all(fake_images[i].flatten() == fake_images[i].flatten()[0])):
                corr = 0.0
            else:
                corr, _ = pearsonr(real_images[i].flatten(), fake_images[i].flatten())
            corr_list.append(corr)

            ssim_score = ssim(real_images[i], fake_images[i], channel_axis=2, data_range=1.0)
            ssim_scores.append(ssim_score)

        corr_mean = np.mean(corr_list)
        ssim_mean = np.mean(ssim_scores)

        print(f"Average Pearson Correlation: {corr_mean:.4f}")
        print(f"Average SSIM: {ssim_mean:.4f}")

        return (total_error / batch_count, total_error_l1 / batch_count,
                total_error_mse / batch_count, corr_mean, ssim_mean)


if __name__ == '__main__':
    stain_panel = read_json_from_txt("./output.txt")

    fixed_stain = ["dapi", "autofluorescence"]
    potential_output = ["cd8", "pd-l1"]

    train_valid_test_data_csv_path = "try_scale.csv"
    results_dir = "./results_SIMIF"

    obj = TrainerCGAN(marker_panel=stain_panel,
                      fixed_markers=fixed_stain,
                      potential_output_markers=potential_output,
                      results_dir=results_dir,
                      target_marker=["cd8"],
                      lr=0.0002, seed=1)

    obj.train(train_valid_test_data_csv_path,
              percent=50, img_size=224,
              batch_size=16, num_workers=4,
              max_epochs=400,
              minimum_epochs=380, patience=5,
              load_model_ckpt=False)
