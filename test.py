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
import os

import cv2
import torch
from natsort import natsorted

import bsrgan_config
import imgproc
import model
from image_quality_assessment import NIQE
from utils import make_directory

model_names = sorted(
    name for name in model.__dict__ if
    name.islower() and not name.startswith("__") and callable(model.__dict__[name]))


def main() -> None:
    # Initialize the super-resolution bsrgan_model
    bsrgan_model = model.__dict__[bsrgan_config.g_arch_name](in_channels=bsrgan_config.in_channels,
                                                             out_channels=bsrgan_config.out_channels,
                                                             channels=bsrgan_config.channels,
                                                             growth_channels=bsrgan_config.growth_channels,
                                                             num_blocks=bsrgan_config.num_blocks)
    bsrgan_model = bsrgan_model.to(device=bsrgan_config.device)
    print(f"Build `{bsrgan_config.g_arch_name}` model successfully.")

    # Load the super-resolution bsrgan_model weights
    checkpoint = torch.load(bsrgan_config.g_model_weights_path, map_location=lambda storage, loc: storage)
    bsrgan_model.load_state_dict(checkpoint["state_dict"])
    print(f"Load `{bsrgan_config.g_arch_name}` model weights "
          f"`{os.path.abspath(bsrgan_config.g_model_weights_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    make_directory(bsrgan_config.sr_dir)

    # Start the verification mode of the bsrgan_model.
    bsrgan_model.eval()

    # Initialize the sharpness evaluation function
    niqe = NIQE(bsrgan_config.upscale_factor, bsrgan_config.niqe_model_path)

    # Set the sharpness evaluation function calculation device to the specified bsrgan_model
    niqe = niqe.to(device=bsrgan_config.device, non_blocking=True)

    # Initialize IQA metrics
    niqe_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(bsrgan_config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    for index in range(total_files):
        sr_image_path = os.path.join(bsrgan_config.sr_dir, file_names[index])
        lr_image_path = os.path.join(bsrgan_config.lr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        lr_tensor = imgproc.preprocess_one_image(lr_image_path, bsrgan_config.device)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            sr_tensor = bsrgan_model(lr_tensor)

        # Save image
        sr_image = imgproc.tensor_to_image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        niqe_metrics += niqe(sr_tensor).item()

    print(f"NIQE: {niqe_metrics / total_files:4.2f} [100u]")


if __name__ == "__main__":
    main()
