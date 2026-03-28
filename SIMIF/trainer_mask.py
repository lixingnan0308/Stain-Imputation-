import json
from math import e
import os
from pyexpat import model
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from captum.attr import IntegratedGradients

from dataloader1 import MxIFReader
#from network_aunet import Generator, Discriminator


from scipy import stats as st
from skimage.metrics import structural_similarity as ssim
import platform

class Trainer:
    def __init__(self, 
                 marker_panel, 
                 #input_markers, 
                 output_markers, 
                 results_dir, lr=0.002, seed=1):
        """
        Trainer class for training and evaluating a protein marker imputation model.

        Args:
            marker_panel (list): A list of marker names in the same order as the channels in the MXIF images.
            input_markers (list): A list of marker names to be used as input to the model.
            output_markers (list): A list of marker names to be used as output to the model.
            results_dir (str): Directory to store the results.
            lr (float, optional): Learning rate for the adam optimizer. Defaults to 0.002.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.
        """
        self.marker_panel = marker_panel
        #self.input_markers = input_markers
        self.pot_output_markers = output_markers
        self.results_dir = results_dir
        self.lr = lr
        self.seed = seed

        self.counter = 0
        self.lowest_loss = np.Inf
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
        Sets the random seed for reproducibility.

        Args:
            seed (int): Random seed.
        """

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            #torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        # MacOS-specific settings
        if platform.system() == 'Darwin':
            torch.use_deterministic_algorithms(True)

    def init_data_loader(self, data_csv_path, percent=100, 
                         img_size=256, batch_size=64, 
                         num_workers=4,
                         input_marker = [],
                         output_marker = []
                         ):
        """
        Initializes the data loader for training and validation data.

        Args:
            data_csv_path (str): Path to the data CSV file.
            percent (int, optional): Percentage of data to use. Defaults to 100.
            img_size (int, optional): Size of the input images. Defaults to 256.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        """
        train_dataset = MxIFReader(data_csv_path=data_csv_path, split_name='train', 
                                        marker_panel=self.marker_panel,
                                        input_markers=input_marker, 
                                        output_markers=output_marker,
                                        training=True, img_size=img_size, percent=percent)
        train_loader = MxIFReader.get_data_loader(train_dataset, batch_size=batch_size, training=True,
                                                       num_workers=num_workers)

        valid_dataset = MxIFReader(data_csv_path=data_csv_path, split_name='valid', 
                                        marker_panel=self.marker_panel,
                                        input_markers=input_marker, 
                                        output_markers=output_marker,
                                        training=False, img_size=img_size)
        valid_loader = MxIFReader.get_data_loader(valid_dataset, batch_size=batch_size, training=False,
                                                       num_workers=num_workers)
        
        return [train_dataset,train_loader,valid_dataset,valid_loader]


    def init_model(self, is_train=False,input_marker=[],output_marker=[],has_d=False):
        """
        Initializes the marker imputation model.

        Args:
            is_train (bool, optional): If True, move the model to one of the gpu if available. Defaults to False.
        """
        self.model_g = Generator(in_channels=len(input_marker), 
                                 out_channels=len(output_marker), 
                                 init_features=32)
        self.model_g = self.model_g.apply(weights_init)
        if is_train:
            self.model_g = self.model_g.to(device=self.device)

    def init_optimizer(self):
        """Initializes the optimizer."""
        self.optimizer = optim.Adam(self.model_g.parameters(), lr=self.lr, betas=(0.5, 0.999))

    def init_loss_function(self):
        """Initializes the loss functions."""
        self.loss_l1 = nn.L1Loss()
        self.loss_mse = nn.MSELoss()

    

    def train(self, data_csv_path, percent=100, img_size=256, batch_size=64, num_workers=4, max_epochs=200,
              minimum_epochs=50, patience=25, load_model_ckpt=False):
        """
        Trains the marker imputation model.

        Args:
            data_csv_path (str): Path to the data CSV file.
            percent (int, optional): Percentage of data to use. Defaults to 100.
            img_size (int, optional): Size of the input images. Defaults to 256.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
            max_epochs (int, optional): Maximum number of epochs. Defaults to 200.
            minimum_epochs (int, optional): Minimum number of epochs before early stopping. Defaults to 50.
            patience (int, optional): Number of epochs to wait for improvement before early stopping. Defaults to 25.

        Returns:
            dict: Dictionary containing training and validation total loss, L1 loss, and MSE loss.
        """
        self.counter = 0
        self.lowest_loss = -np.Inf
        self.set_seed(seed=self.seed)
        self.branch_loaders = []
        self.branch_models_g =[]
        self.optimizers = []
        self.losses = []
        self.has_discriminator =False
        self.has_discriminator_optim = False
        self.load_model_ckpt = load_model_ckpt
        self.stain_indexes = []
        for index, potential_output in enumerate(self.pot_output_markers):
            input_marker = self.marker_panel.copy()
            input_marker.remove(potential_output)
            output_marker = potential_output
            if output_marker == "cd8":
                input_marker.remove("pd-l1")
            elif output_marker == "pd-l1":
                input_marker.remove("cd8")
            else:
                raise ValueError('No Target Output Markers Found')

            left_pot_marker = input_marker.copy()
            left_pot_marker.remove("dapi")
            left_pot_marker.remove("autofluorescence")
            stain_index = [input_marker.index(element) for element in input_marker if element in left_pot_marker]
            print(input_marker,stain_index)
            self.stain_indexes.append(stain_index)
            
            branch_loader = self.init_data_loader(data_csv_path, 
                                                       percent=percent, 
                                                       img_size=img_size, 
                                                       batch_size=batch_size, 
                                                       num_workers=num_workers,
                                                       input_marker = input_marker,
                                                       output_marker = [output_marker]
                                                       )
            
            self.branch_loaders.append(branch_loader)
            
            if not self.has_discriminator:
                branch_model_g , self.model_d= self.init_model(is_train=True,
                                                input_marker = input_marker,
                                                output_marker = [output_marker],
                                                had_d = self.has_discriminator
                                                )
                
                if platform.system() == 'Darwin':
                    branch_model_g = branch_model_g.to(dtype=torch.bfloat16)
                    self.model_d = self.model_d.to(dtype=torch.bfloat16)

                if self.load_model_ckpt:
                    branch_model_g = self.load_model(ckpt_path=os.path.join(self.results_dir, 'checkpoint_130.pt'),
                                                     model_g = branch_model_g, index = index)
                    if torch.cuda.is_available():
                        branch_model_g = branch_model_g.to(device=self.device)
                    else:
                        branch_model_g = branch_model_g.to(device=self.device).to(dtype=torch.bfloat16)
                    self.load_model_d(ckpt_path=os.path.join(self.results_dir, 'checkpoint_d_130.pt'))
                self.branch_models_g.append(branch_model_g)
                self.model_d = self.model_d.to(device=self.device)
                self.has_discriminator = True
            else:
                branch_model_g = self.init_model(is_train=True,
                                                input_marker = input_marker,
                                                output_marker = [output_marker],
                                                had_d = self.has_discriminator
                                                )
                if platform.system() == 'Darwin':
                    branch_model_g = branch_model_g.to(dtype=torch.bfloat16)
            
                if self.load_model_ckpt:
                    branch_model_g = self.load_model(ckpt_path=os.path.join(self.results_dir, 'checkpoint_130.pt'),model_g = branch_model_g, index = index)
                    if torch.cuda.is_available():
                        branch_model_g = branch_model_g.to(device=self.device)
                    else:
                        branch_model_g = branch_model_g.to(device=self.device).to(dtype=torch.bfloat16)
                    self.load_model_d(ckpt_path=os.path.join(self.results_dir, 'checkpoint_d_130.pt'))
                self.branch_models_g.append(branch_model_g)
        
            if not self.has_discriminator_optim:
                optimizer ,self.optimizer_d= self.init_optimizer(model_g = branch_model_g,
                                    model_d = self.model_d, 
                                    has_o = self.has_discriminator_optim)
                
                self.optimizers.append(optimizer)
                self.has_discriminator_optim = True
                                   
            else:
                optimizer = self.init_optimizer(model_g = branch_model_g,
                                    model_d = self.model_d,
                                    has_o = self.has_discriminator_optim)
                self.optimizers.append(optimizer)

        self.init_loss_function()

        result_dict = {'train_loss': [], 'valid_loss': [], 'train_l1': [], 'valid_l1': [], 'train_mse': [], 'valid_mse': [],"valid":[]}
        for epoch in range(max_epochs):
            start_time = time.time()

            train_loss = []
            train_l1 = []
            train_mse = []
            
            for index, loader in enumerate(self.branch_loaders):   
                train_loss_b, train_l1_b, train_mse_b = self.train_loop(loader[1], index, epoch)
                
                train_loss.append(train_loss_b)
                train_l1.append(train_l1_b)
                train_mse.append(train_mse_b)
                break
                
                
            train_loss = np.mean(train_loss)
            train_l1 = np.mean(train_l1)
            train_mse = np.mean(train_mse)

            result_dict['train_loss'].append(train_loss)
            result_dict['train_l1'].append(train_l1)
            result_dict['train_mse'].append(train_mse)

            print('\rTrain Epoch: {}, train_loss: {:.4f}, train_l1: {:.4f}, train_mse: {:.4f}      '.format(epoch, 
                                                                                                            train_loss, train_l1, train_mse))

            valid_loss = []
            valid_l1 = []
            valid_mse = []
            valid_corr = []
            valid_ssim = []
            for index, loader in enumerate(self.branch_loaders #self.branch_loaders[1]
                                           ):
                
                valid_loss_b, valid_l1_b, valid_mse_b ,valid_corr_b, valid_ssim_b= self.valid_loop(loader[-1],index,use_mask=True,epoch=epoch)

                valid_loss.append(valid_loss_b)
                valid_l1.append(valid_l1_b)
                valid_mse.append(valid_mse_b)
                valid_corr.append(valid_corr_b)
                valid_ssim.append(valid_ssim_b)
                break
                
                
                
            
            valid_loss = np.mean(valid_loss)
            valid_l1 = np.mean(valid_l1)
            valid_mse = np.mean(valid_mse)
            valid_corr = np.mean(valid_corr)
            valid_ssim = np.mean(valid_ssim)

            valid = 0.5*valid_corr + 0.5*valid_ssim

            result_dict['valid_loss'].append(valid_loss)
            result_dict['valid_l1'].append(valid_l1)
            result_dict['valid_mse'].append(valid_mse)
            result_dict['valid'].append(valid)

            print('\rValid Epoch: {}, valid_loss: {:.4f}, valid_l1: {:.4f}, valid_mse: {:.4f}     '.format(epoch, 
                                                                                                           valid_loss, valid_l1, valid_mse))
            print(valid)
            print("\n")   

            if self.lowest_loss < valid or epoch%10 ==0:
                model_params_dict = {}
                for idx, model_param in enumerate(self.branch_models_g):
                    model_params_dict[f'model_param_{idx}'] = model_param.state_dict()
                
                print('--------------------Saving best model--------------------')
                check_point = f"checkpoint_{epoch}.pt"
                check_point_d = f"checkpoint_d_{epoch}.pt"
                torch.save(model_params_dict, os.path.join(self.results_dir, 
                                                                check_point))
                torch.save(self.model_d.state_dict(), os.path.join(self.results_dir, check_point_d))
                self.lowest_loss = valid
                self.counter = 0
            else:
                self.counter += 1
                print('Loss is not decreased in last %d epochs' % self.counter)

            if (self.counter > patience) and (epoch >= minimum_epochs):
                break

            total_time = time.time() - start_time
            print('Time to process epoch({}): {:.4f} minutes                             \n'.format(epoch, total_time/60))
            pd.DataFrame.from_dict(result_dict).to_csv(os.path.join(self.results_dir, 'training_stats.csv'), index=False)
        return result_dict

    def train_loop(self, data_loader):
        """
        Training loop for a single epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the training data.

        Returns:
            float: Average training loss.
            float: Average L1 loss.
            float: Average MSE loss.
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

            print('Training - [%d/%d]\tL1 Loss: %.06f \tMSE Loss: %.06f                                                            '
                  % (batch_idx, batch_count, error_l1.item(), error_mse.item()), end='\r')
            total_error += error.item()
            total_error_l1 += error_l1.item()
            total_error_mse += error_mse.item()


        return total_error / batch_count, total_error_l1 / batch_count, total_error_mse / batch_count

    def valid_loop(self, data_loader):
        """
        Validation loop for a single epoch.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the validation data.

        Returns:
            float: Average validation loss.
            float: Average L1 loss.
            float: Average MSE loss.
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

                print('Validation - [%d/%d]\tL1 Loss: %.06f \tMSE Loss: %.06f                      '
                    % (batch_idx, batch_count, error_l1.item(), error_mse.item()), end='\r')
                total_error += error.item()
                total_error_l1 += error_l1.item()
                total_error_mse += error_mse.item()


        return total_error / batch_count, total_error_l1 / batch_count, total_error_mse / batch_count

    
    def load_model(self, ckpt_path,model_g, index):
        """
        Loads the model from a checkpoint file.

        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        all_check_point = torch.load(ckpt_path,map_location=torch.device('cpu'))
        ckpt = all_check_point[f'model_param_{index}']
        ckpt_clean = {}
        for key in ckpt.keys():
            ckpt_clean.update({key.replace('module.', ''): ckpt[key]})
        model_g.load_state_dict(ckpt_clean, strict=True)
        model_g = model_g.to(device=self.device)

        return model_g
    
    def load_model_d(self, ckpt_path):
        """
        Loads the model from a checkpoint file.

        Args:
            ckpt_path (str): Path to the checkpoint file.
        """
        ckpt = torch.load(ckpt_path)
        ckpt_clean = {}
        for key in ckpt.keys():
            ckpt_clean.update({key.replace('module.', ''): ckpt[key]})
        self.model_d.load_state_dict(ckpt_clean, strict=True)
        if torch.cuda.is_available():
            self.model_d = self.model_d.to(device=self.device)
        else:
            self.model_d = self.model_d.to(device=self.device).to(dtype=torch.bfloat16)

    
    
    def eval(self, data_csv_path, split_name='test', img_size=256, batch_size=64, num_workers=4,required_stains=[]):
        """
        Evaluates the trained model on the test data.

        Args:
            data_csv_path (str): Path to the data CSV file.
            split_name (str, optional): Name of the data split to evaluate. Defaults to 'test'.
            img_size (int, optional): Size of the input images. Defaults to 256.
            batch_size (int, optional): Batch size. Defaults to 64.
            num_workers (int, optional): Number of workers for data loading. Defaults to 4.
        """
        
        data_loaders = []
        indexs = []
        model_gs = []
        for id, stain in enumerate(required_stains):
            index = self.pot_output_markers.index(stain)
            input_markers = self.marker_panel.copy()
            input_markers.remove(stain)
            output_markers = [stain]
            self.set_seed(self.seed)

            left_pot_marker = self.pot_output_markers.copy()
            left_pot_marker.remove(stain)
            stain_index= [input_markers.index(element) for element in left_pot_marker if element in input_markers]
            self.stain_indexes.append(stain_index)

            dataset = MxIFReader(data_csv_path=data_csv_path, split_name=split_name, marker_panel=self.marker_panel,
                                            input_markers=input_markers, output_markers=output_markers,
                                            training=False, img_size=img_size,percent=20)
            data_loader = MxIFReader.get_data_loader(dataset, batch_size=batch_size, training=False, num_workers=num_workers)

        
            model_g = self.init_model(is_train=False,input_marker=input_markers,
                                      output_marker=[output_markers], had_d=True)
            # if platform.system() == 'Darwin':
            #     model_g = model_g.to(dtype=torch.bfloat16)
            checkpoint_path = os.path.join(self.results_dir, 'checkpoint.pt')
            
            
            model_g = self.load_model(ckpt_path=os.path.join(self.results_dir, 'checkpoint_mask220.pt'),
                                                             model_g = model_g, 
                                                             index = id)
            eval_dir_name = '%s_%d_%d' % (split_name+"_1mask220", img_size, img_size)
            data_loaders.append(data_loader)
            indexs.append(index)
            model_gs.append(model_g)

        self.eval_loop(data_loaders, eval_dir_name, model_gs, indexs)
        
    def mask_input_batchs(self, input_batch, stain_index,numbers=1):
        """
        Masks the input batch by setting the channels corresponding to the output markers to zero.

        Args:
            input_batch (torch.Tensor): Input batch of images. #(B, C, H, W)
            stain_index (list): List of indexes for the output markers.

        Returns:
            torch.Tensor: Masked input batch.
        """
        
        
        numbers = random.sample(stain_index, numbers)
        if len(numbers) == 0:
            masked_input_batch = input_batch.clone()
            return masked_input_batch
        else:
            masked_input_batch = input_batch.clone()
            for num in numbers:
                masked_input_batch[:, num, :, :] = 0
        return masked_input_batch        

    def eval_loop(self, data_loaders, eval_dir_name, model_gs, indexs):
        """
        Evaluation loop for the test data.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the test data.
            eval_dir_name (str): Name of the directory to save the evaluation results.
        """
        stats_dict = {"Image_Name":[], "Stain":[], "MAE":[], "MSE": [], "SSIM": [], "PSNR": [], "RMSE": [], "Corr": [], "p-value": []}
        stats_dict_zeros = {"Image_Name": [], "Stain":[],"MAE": [], "MSE": [], "SSIM": [], "PSNR": [], "RMSE": []}
        stats_dict_mean = {"Image_Name": [], "Stain":[],"MAE": [], "MSE": [], "SSIM": [], "PSNR": [], "RMSE": []}

        
        for id, data_loader in enumerate(data_loaders):
            model_g = model_gs[id].to(torch.float32).to(self.device)
            index = indexs[id]
            stain_name = self.pot_output_markers[index]
            batch_count = len(data_loader)
            model_g.eval()

            stain_index = self.stain_indexes[id]
            
            os.makedirs(os.path.join(self.results_dir, eval_dir_name, stain_name), exist_ok=True)

            with torch.no_grad():
                for batch_idx, (input_batch, output_batch, image_name_batch, img_dims) in enumerate(data_loader):
                    input_batch = self.mask_input_batch(input_batch, stain_index, 1)
                    input_batch = input_batch.to(self.device).to(torch.float32)
                    output_batch = output_batch.to(self.device).to(torch.float32)
                    
                    generated_batch = model_g(input_batch) ###



                    for i, image_name in enumerate(image_name_batch):
                        img_dim = [img_dims[0][i].item(), img_dims[1][i].item()]
                        image_name = os.path.basename(image_name)
                        image_name, ext = os.path.splitext(image_name)

                        print('%d/%d - (%d) %s' % (batch_idx, batch_count, i, image_name))
                        input = input_batch[i, :, :, :].detach().cpu().numpy()
                        real = output_batch[i, :, :, :].detach().cpu().numpy()
                        generated = generated_batch[i, :, :, :].detach().cpu().numpy()

                        input = input[:, :img_dim[0], :img_dim[1]]
                        real = real[:, :img_dim[0], :img_dim[1]]
                        generated = generated[:, :img_dim[0], :img_dim[1]]


                        output = np.concatenate([real, generated], axis=0)
                        np.save(os.path.join(self.results_dir, eval_dir_name, stain_name, image_name + '.npy'), output)
                        
                        input = input * 255.0
                        real = real * 255.0
                        generated = generated * 255.0

                        zero_image = np.zeros_like(real)
                        mean_image = np.mean(input, axis=0)

                        stats = self.pixel_metrics(real, generated, max_val=255, baseline=False)
                        stats_zeros = self.pixel_metrics(real, zero_image, max_val=255, baseline=True)
                        stats_mean = self.pixel_metrics(real, mean_image, max_val=255, baseline=True)

                        stats_dict["Image_Name"].append(image_name)
                        stats_dict["Stain"].append(stain_name)
                        stats_dict_zeros["Image_Name"].append(image_name)
                        stats_dict_zeros["Stain"].append(stain_name)
                        stats_dict_mean["Image_Name"].append(image_name)
                        stats_dict_mean["Stain"].append(stain_name)

                        for key in stats.keys():
                            stats_dict[key].append(stats[key])
                        for key in stats_zeros.keys():
                            stats_dict_zeros[key].append(stats_zeros[key])
                        for key in stats_mean.keys():
                            stats_dict_mean[key].append(stats_mean[key])
        
        pd.DataFrame.from_dict(stats_dict).to_csv(os.path.join(self.results_dir, '%s_stats.csv' % eval_dir_name), index=False)
        pd.DataFrame.from_dict(stats_dict_zeros).to_csv(os.path.join(self.results_dir, '%s_stats_zero.csv' % eval_dir_name), index=False)
        pd.DataFrame.from_dict(stats_dict_mean).to_csv(os.path.join(self.results_dir, '%s_stats_mean.csv' % eval_dir_name), index=False)

    @staticmethod
    def pixel_metrics(real, generated, max_val=255, baseline=False):
        """
        Computes pixel-level evaluation metrics between the ground truth and generated images.

        Args:
            real (numpy.ndarray): Ground truth images.
            generated (numpy.ndarray): Generated images.
            max_val (float, optional): Maximum pixel value. Defaults to 255.
            baseline (bool, optional): If True, compares against a baseline image. Defaults to False.

        Returns:
            dict: Dictionary containing pixel-level evaluation metrics.
        """
        real= np.squeeze(real)
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

    def attributions(self, data_csv_path, split_name='test', img_size=256, batch_size=32, num_workers=4):
        """
         Computes and saves attributions for the test data.

         Args:
             data_csv_path (str): Path to the data CSV file.
             split_name (str, optional): Name of the data split to evaluate. Defaults to 'test'.
             img_size (int, optional): Size of the input images. Defaults to 256.
             batch_size (int, optional): Batch size. Defaults to 32.
             num_workers (int, optional): Number of workers for data loading. Defaults to 4.
         """
        self.set_seed(self.seed)
        dataset = MxIFReader(data_csv_path=data_csv_path, split_name=split_name, marker_panel=self.marker_panel,
                                        input_markers=self.input_markers, output_markers=self.output_markers,
                                        training=False, img_size=img_size)
        data_loader = MxIFReader.get_data_loader(dataset, batch_size=batch_size, training=False, num_workers=num_workers)

        self.init_model()
        self.load_model(ckpt_path=os.path.join(self.results_dir, 'checkpoint.pt'))
        attr_dir_name = 'attributions_%s_%d_%d' % (split_name, img_size, img_size)
        self.attributions_loop(data_loader, attr_dir_name)

    def attributions_loop(self, data_loader, attr_dir_name):
        """
        Attribution computation loop for the test data.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for the test data.
            attr_dir_name (str): Name of the directory to save the attributions.
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
            attr, _ = ig.attribute(input_batch, baselines=torch.zeros_like(input_batch, device=self.device), target=0,
                                   return_convergence_delta=True)

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
                neg_attr_ = np.maximum((-1)*attr_, 0)
                attr_ = np.expand_dims(np.sum(np.sum(np.abs(attr_), axis=-1), axis=-1), axis=0)
                pos_attr_ = np.expand_dims(np.sum(np.sum(pos_attr_, axis=-1), axis=-1), axis=0)
                neg_attr_ = np.expand_dims(np.sum(np.sum(neg_attr_, axis=-1), axis=-1), axis=0)

                if attr_array is None:
                    attr_array_pos = pos_attr_
                    attr_array_neg = neg_attr_
                    attr_array = attr_
                else:
                    attr_array = np.concatenate((attr_array,attr_), axis=0)
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
        pred = self.model_g(batch)
        pred = nn.AdaptiveAvgPool2d((1,1))(pred)
        return pred

def read_json_from_txt(file_path):
    with open(file_path, "r") as file:
        data = file.read()

    # load json as dict
    original_dict = json.loads(data)
    values_list = []
    for value in original_dict.values():
        value =value.split(' ')[0]
        values_list.append(value)

    return values_list



if __name__ == '__main__':
    """
    marker_panel = ['DAPI', 'FOXP3', 'CD4', 'CD8', 'PDL1', 'KI67', 'CK']
    input_markers = ['DAPI', 'FOXP3', 'CD4', 'CD8', 'PDL1', 'CK']
    output_markers = ['KI67']
    train_valid_test_data_csv_path = '/media/shaban/hd1/Projects_HD1/TIME/Marker_Synthesis/datasets/panel1/train_valid_test_patches_ssd.csv' 
    inference_data_csv_path = '/media/shaban/hd1/Projects_HD1/TIME/Marker_Synthesis/datasets/panel1/test_images.csv'
    results_dir = '/media/shaban/hd1/Projects_HD1/TIME/Marker_Synthesis/results/temp_results/'

    """
    stain_panel = read_json_from_txt("./output.txt")
    potential_output = stain_panel.copy()
    fixed_stain = ["dapi", "autofluorescence"]
    potential_output = [x for x in potential_output if x not in fixed_stain]

    train_valid_test_data_csv_path = "./new_path_split_36.csv"
    inference_data_csv_path = None
    results_dir = "./results/36_results/"

    obj = Trainer(marker_panel=stain_panel,
                  #input_markers=input_stain_panel,
                  output_markers=potential_output,
                  results_dir=results_dir,
                  lr=0.002, seed=1)

    obj.train(train_valid_test_data_csv_path,
              percent=30, img_size=224,
              batch_size=64, num_workers=4,
              max_epochs=100,
              minimum_epochs=80, patience=1)

    
