# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
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
import random

import numpy as np
import torch
from torch.backends import cudnn

degradation_process_parameters_dict = {
    "jpeg_prob": 0.9,
    "scale2_prob": 0.25,
    "shuffle_prob": 0.1,
    "use_sharp": False,
}

degradation_process_plus_parameters_dict = {
    "shuffle_prob": 0.1,
    "poisson_prob": 0.1,
    "speckle_prob": 0.1,
    "use_sharp": True,
}

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# When evaluating the performance of the SR model, whether to verify only the Y channel image data
only_test_y_channel = True
# NIQE model address
niqe_model_path = "./results/pretrained_models/niqe_model.mat"
# Model architecture name
g_model_arch_name = "bsrgan_x4"
# Model arch config
g_in_channels = 3
g_out_channels = 3
g_channels = 64
g_growth_channels = 32
g_num_rrdb = 23
# Upscale factor
upscale_factor = 4
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
exp_name = "BSRNet_x4-DIV2K"

if mode == "train":
    # Dataset address
    train_gt_images_dir = f"./data/DIV2K/BSRGAN/train"

    test_gt_images_dir = f"./data/Set5/GTmod12"
    test_lr_images_dir = f"./data/Set5/LRbicx{upscale_factor}"

    crop_image_size = 320
    gt_image_size = int(72 * upscale_factor)
    batch_size = 16
    num_workers = 4

    # Load the address of the pretrained model
    pretrained_g_model_weights_path = "./results/pretrained_models/BSRNet_x4-DIV2K-353eb572.pth.tar"

    # Incremental training and migration training
    resume_g_model_weights_path = ""

    # Total num epochs (2,000,000 iters)
    epochs = 2050

    # loss function weights
    loss_weight = [1.0]

    # Optimizer parameter
    model_lr = 1e-4
    model_betas = (0.9, 0.99)
    model_eps = 1e-4  # Keep no nan
    model_weight_decay = 0.0

    # EMA parameter
    model_ema_decay = 0.999

    # Dynamically adjust the learning rate policy
    lr_scheduler_milestones = [int(epochs * 0.1), int(epochs * 0.2), int(epochs * 0.3), int(epochs * 0.4), int(epochs * 0.5)]
    lr_scheduler_gamma = 0.5

    # How many iterations to print the training result
    train_print_frequency = 100
    test_print_frequency = 1

if mode == "test":
    # Test data address
    lr_dir = "./data/RealSRSet"
    sr_dir = f"./results/{exp_name}"

    g_model_weights_path = "./results/pretrained_models/BSRNet_x4-DIV2K-353eb572.pth.tar"
