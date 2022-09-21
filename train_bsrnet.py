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
from torch import nn
from torch import optim
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
from utils import load_state_dict, make_directory, save_checkpoint, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_psnr = 0.0
    best_ssim = 0.0

    train_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")

    bsrnet_model, ema_bsrnet_model = build_model()
    print(f"Build `{bsrnet_config.g_arch_name}` model successfully.")

    criterion = define_loss()
    print("Define all loss functions successfully.")

    optimizer = define_optimizer(bsrnet_model)
    print("Define all optimizer functions successfully.")

    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")

    print("Check whether to load pretrained model weights...")
    if bsrnet_config.pretrained_g_model_weights_path:
        bsrnet_model = load_state_dict(bsrnet_model, bsrnet_config.pretrained_g_model_weights_path)
        print(f"Loaded `{bsrnet_config.pretrained_g_model_weights_path}` pretrained model weights successfully.")
    else:
        print("Pretrained model weights not found.")

    print("Check whether the pretrained model is restored...")
    if bsrnet_config.resume_g:
        bsrnet_model, ema_bsrnet_model, start_epoch, est_psnr, best_ssim, optimizer, scheduler = load_state_dict(
            bsrnet_model,
            bsrnet_config.pretrained_g_model_weights_path,
            ema_bsrnet_model,
            optimizer,
            scheduler,
            "resume")
        print("Loaded pretrained model weights.")
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

    # Transfer the IQA model to the specified device
    psnr_model = psnr_model.to(device=bsrnet_config.device)
    ssim_model = ssim_model.to(device=bsrnet_config.device)

    for epoch in range(start_epoch, bsrnet_config.epochs):
        train(bsrnet_model,
              ema_bsrnet_model,
              train_prefetcher,
              criterion,
              optimizer,
              epoch,
              scaler,
              writer)
        psnr, ssim = validate(bsrnet_model,
                              test_prefetcher,
                              epoch,
                              writer,
                              psnr_model,
                              ssim_model,
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
                         "state_dict": bsrnet_model.state_dict(),
                         "ema_state_dict": ema_bsrnet_model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict()},
                        f"epoch_{epoch + 1}.pth.tar",
                        samples_dir,
                        results_dir,
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
    bsrnet_model = model.__dict__[bsrnet_config.g_arch_name](in_channels=bsrnet_config.in_channels,
                                                             out_channels=bsrnet_config.out_channels,
                                                             channels=bsrnet_config.channels,
                                                             growth_channels=bsrnet_config.growth_channels,
                                                             num_blocks=bsrnet_config.num_blocks)
    bsrnet_model = bsrnet_model.to(device=bsrnet_config.device)

    # Create an Exponential Moving Average Model
    ema_avg = lambda averaged_model_parameter, model_parameter, num_averaged: (1 - bsrnet_config.model_ema_decay) * averaged_model_parameter + bsrnet_config.model_ema_decay * model_parameter
    ema_bsrnet_model = AveragedModel(bsrnet_model, avg_fn=ema_avg)

    return bsrnet_model, ema_bsrnet_model


def define_loss() -> nn.L1Loss:
    criterion = nn.L1Loss()
    criterion = criterion.to(device=bsrnet_config.device)

    return criterion


def define_optimizer(bsrnet_model) -> optim.Adam:
    optimizer = optim.Adam(bsrnet_model.parameters(),
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
        bsrnet_model: nn.Module,
        ema_bsrnet_model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.L1Loss,
        optimizer: optim.Adam,
        epoch: int,
        scaler: amp.GradScaler,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    bsrnet_model.train()

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
        gt = batch_data["gt"].to(device=bsrnet_config.device, non_blocking=True)
        lr = batch_data["lr"].to(device=bsrnet_config.device, non_blocking=True)

        # Clamp and round
        gt = torch.clamp((gt * 255.0).round(), 0, 255) / 255.
        lr = torch.clamp((lr * 255.0).round(), 0, 255) / 255.

        # Crop image patch
        gt, lr = random_crop(gt, lr, bsrnet_config.gt_image_size, bsrnet_config.upscale_factor)

        # Initialize generator gradients
        bsrnet_model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = bsrnet_model(lr)
            loss = torch.mul(bsrnet_config.loss_weights, criterion(sr, gt))

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_bsrnet_model.update_parameters(bsrnet_model)

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % bsrnet_config.train_print_frequency == 0:
            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        bsrnet_model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        psnr_model: nn.Module,
        ssim_model: nn.Module,
        mode: str
) -> [float, float]:
    # Calculate how many batches of data are in each Epoch
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(len(data_prefetcher), [batch_time, psnres, ssimes], prefix=f"{mode}: ")

    # Put the adversarial network model in validation mode
    bsrnet_model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer the in-memory data to the CUDA device to speed up the test
            gt = batch_data["gt"].to(device=bsrnet_config.device, non_blocking=True)
            lr = batch_data["lr"].to(device=bsrnet_config.device, non_blocking=True)

            # Use the generator model to generate a fake sample
            with amp.autocast():
                sr = bsrnet_model(lr)

            # Statistical loss value for terminal data output
            psnr = psnr_model(sr, gt)
            ssim = ssim_model(sr, gt)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % bsrnet_config.valid_print_frequency == 0:
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    main()
