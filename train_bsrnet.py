# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time

import torch
from torch import nn, optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import bsrnet_config
import model
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from image_quality_assessment import PSNR, SSIM
from imgproc import random_crop
from utils import load_state_dict, make_directory, save_checkpoint, validate, AverageMeter, ProgressMeter


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    g_model, ema_g_model = build_model()
    print(f"Build `{bsrnet_config.g_model_arch_name}` model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(g_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if bsrnet_config.pretrained_g_model_weights_path:
        g_model = load_state_dict(g_model, bsrnet_config.pretrained_g_model_weights_path)
        print(f"Loaded `{bsrnet_config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the resume model is restored...")
    if bsrnet_config.resume_g_model_weights_path:
        g_model, ema_g_model, start_epoch, best_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            g_model,
            bsrnet_config.pretrained_g_model_weights_path,
            ema_g_model,
            optimizer,
            scheduler,
            "resume")
        print("Loaded resume model weights.")
    else:
        print("Resume training model not found. Start training from scratch.")

    # Create a experiment results
    samples_dir = os.path.join("samples", bsrnet_config.exp_name)
    results_dir = os.path.join("results", bsrnet_config.exp_name)
    make_directory(samples_dir)
    make_directory(results_dir)

    # Create training process log file
    writer = SummaryWriter(os.path.join("samples", "logs", bsrnet_config.exp_name))

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    psnr_model = PSNR(bsrnet_config.upscale_factor, bsrnet_config.only_test_y_channel)
    ssim_model = SSIM(bsrnet_config.upscale_factor, bsrnet_config.only_test_y_channel)
    psnr_model = psnr_model.to(device=bsrnet_config.device)
    ssim_model = ssim_model.to(device=bsrnet_config.device)

    for epoch in range(start_epoch, bsrnet_config.epochs):
        train(g_model,
              ema_g_model,
              train_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              writer,
              bsrnet_config.device,
              bsrnet_config.train_print_frequency)
        psnr, ssim = validate(g_model,
                              test_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
                              bsrnet_config.device,
                              bsrnet_config.test_print_frequency,
                              "Test")
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = psnr > best_psnr and ssim > best_ssim
        is_last = (epoch + 1) == bsrnet_config.epochs
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        save_checkpoint({"epoch": epoch + 1,
                         "best_psnr": best_psnr,
                         "best_ssim": best_ssim,
                         "state_dict": g_model.state_dict(),
                         "ema_state_dict": ema_g_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"g_epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
                        "g_best.pth.tar",
                        "g_last.pth.tar",
                        is_best,
                        is_last)


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher]:
    # Load train, test and valid datasets
    train_datasets = TrainValidImageDataset(bsrnet_config.train_gt_images_dir,
                                            bsrnet_config.crop_image_size,
                                            bsrnet_config.upscale_factor,
                                            "Train",
                                            bsrnet_config.degradation_process_parameters_dict)
    test_datasets = TestImageDataset(bsrnet_config.test_gt_images_dir, bsrnet_config.test_lr_images_dir)

    # Generator all dataloader
    train_dataloader = DataLoader(train_datasets,
                                  batch_size=bsrnet_config.batch_size,
                                  shuffle=True,
                                  num_workers=bsrnet_config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_datasets,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 drop_last=False,
                                 persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, bsrnet_config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, bsrnet_config.device)

    return train_prefetcher, test_prefetcher


def build_model() -> [nn.Module, nn.Module]:
    g_model = model.__dict__[bsrnet_config.g_model_arch_name](in_channels=bsrnet_config.g_in_channels,
                                                              out_channels=bsrnet_config.g_out_channels,
                                                              channels=bsrnet_config.g_channels,
                                                              growth_channels=bsrnet_config.g_growth_channels,
                                                              num_rrdb=bsrnet_config.g_num_rrdb)
    g_model = g_model.to(device=bsrnet_config.device)

    # Create an Exponential Moving Average Model
    ema_avg_fn = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - bsrnet_config.model_ema_decay) * averaged_model_parameter + bsrnet_config.model_ema_decay * model_parameter
    ema_g_model = AveragedModel(g_model, avg_fn=ema_avg_fn)

    return g_model, ema_g_model


def define_loss() -> nn.L1Loss:
    criterion = nn.L1Loss()
    criterion = criterion.to(device=bsrnet_config.device)

    return criterion


def define_optimizer(g_model: nn.Module) -> optim.Adam:
    optimizer = optim.Adam(g_model.parameters(),
                           bsrnet_config.model_lr,
                           bsrnet_config.model_betas,
                           bsrnet_config.model_eps)

    return optimizer


def define_scheduler(optimizer: optim.Adam) -> lr_scheduler.MultiStepLR:
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         bsrnet_config.lr_scheduler_milestones,
                                         bsrnet_config.lr_scheduler_gamma)

    return scheduler


def train(
        g_model: nn.Module,
        ema_g_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.L1Loss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter,
        device: torch.device = torch.device("cpu"),
        print_frequency: int = 1,
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    g_model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        gt = batch_data["gt"].to(device=device, non_blocking=True)
        lr = batch_data["lr"].to(device=device, non_blocking=True)
        loss_weight = torch.Tensor(bsrnet_config.loss_weight).to(device=device)

        # Crop image patch
        gt, lr = random_crop(gt, lr, bsrnet_config.gt_image_size, bsrnet_config.upscale_factor)

        # Initialize generator gradients
        g_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = g_model(lr)
            loss = criterion(sr, gt)
            loss = torch.sum(torch.mul(loss_weight, loss))

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_g_model.update_parameters(g_model)

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


if __name__ == "__main__":
    main()
