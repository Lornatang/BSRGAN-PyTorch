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
import math
import random
from typing import Any

import cv2
import numpy as np
import scipy
import torch
from numpy import ndarray
from scipy import ndimage
from scipy.interpolate import interp2d
from scipy.linalg import orth
from scipy.stats import multivariate_normal
from torch import Tensor

__all__ = [
    "image_to_tensor", "tensor_to_image",
    "image_resize", "preprocess_one_image", "degradation_process", "degradation_process_plus",
    "expand_y", "rgb_to_ycbcr", "bgr_to_ycbcr", "ycbcr_to_bgr", "ycbcr_to_rgb",
    "rgb_to_ycbcr_torch", "bgr_to_ycbcr_torch",
    "random_crop_np",
    "center_crop", "random_crop", "random_rotate", "random_vertically_flip", "random_horizontally_flip",
]


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _cubic(x: Any) -> Any:
    """Implementation of `cubic` function in Matlab under Python language.

    Args:
        x: Element vector.

    Returns:
        Bicubic interpolation

    """
    absx = torch.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * ((absx <= 1).type_as(absx)) + (
            -0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (
               ((absx > 1) * (absx <= 2)).type_as(absx))


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _calculate_weights_indices(in_length: int,
                               out_length: int,
                               scale: float,
                               kernel_width: int,
                               antialiasing: bool) -> [np.ndarray, np.ndarray, int, int]:
    """Implementation of `calculate_weights_indices` function in Matlab under Python language.

    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in PIL uses antialiasing by default.

    Returns:
       weights, indices, sym_len_s, sym_len_e

    """
    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialiasing
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _shift_pixel(x, sf, upper_left=True):
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    if x.ndim == 2:
        x = interp2d(xv, yv, x)(x1, y1)
    if x.ndim == 3:
        for i in range(x.shape[-1]):
            x[:, :, i] = interp2d(xv, yv, x[:, :, i])(x1, y1)

    return x


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _fspecial_gaussian(hsize: int, sigma: float):
    hsize = [hsize, hsize]
    size = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-size[1], size[1] + 1), np.arange(-size[0], size[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _fspecial_laplacian(alpha):
    alpha = max([0, min([alpha, 1])])
    h1 = alpha / (alpha + 1)
    h2 = (1 - alpha) / (alpha + 1)
    h = [[h1, h2, h1], [h2, -4 / (alpha + 1), h2], [h1, h2, h1]]
    h = np.array(h)
    return h


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _gm_blur_kernel(mean: Any, cov: Any, size: int) -> ndarray:
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = multivariate_normal.pdf([cx, cy], mean, cov)

    k = k / np.sum(k)

    return k


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _anisotropic_gaussian(ksize: int = 15, theta: float = np.pi, l1: int | float = 6, l2: int | float = 6) -> ndarray:
    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = _gm_blur_kernel([0, 0], Sigma, ksize)

    return k


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _add_blur(image: ndarray, upscale_factor: int = 4) -> ndarray:
    wd = 2.0 + 0.2 * upscale_factor
    wd2 = 4.0 + upscale_factor

    if random.random() < 0.5:
        l1 = wd2 * random.random()
        l2 = wd2 * random.random()
        k = _anisotropic_gaussian(2 * random.randint(2, 11) + 3, random.random() * np.pi, l1, l2)
    else:
        k = _fspecial_gaussian(2 * random.randint(2, 11) + 3, wd * random.random())

    image = ndimage.filters.convolve(image, np.expand_dims(k, axis=2), mode='mirror')

    return image


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _add_gaussian_noise(image: ndarray, noise_level1: int = 2, noise_level2: int = 25) -> ndarray:
    noise_level = random.randint(noise_level1, noise_level2)
    rnum = np.random.rand()
    if rnum > 0.6:  # Add color Gaussian noise
        image += np.random.normal(0, noise_level / 255.0, image.shape).astype(np.float32)
    elif rnum < 0.4:  # Add grayscale Gaussian noise
        image += np.random.normal(0, noise_level / 255.0, (*image.shape[:2], 1)).astype(np.float32)
    else:  # add  noise
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        image += np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), image.shape[:2]).astype(np.float32)
    image = np.clip(image, 0.0, 1.0)
    return image


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _add_poisson_noise(image: ndarray):
    image = np.clip((image * 255.0).round(), 0, 255) / 255.
    vals = 10 ** (2 * random.random() + 2.0)  # [2, 4]
    if random.random() < 0.5:
        image = np.random.poisson(image * vals).astype(np.float32) / vals
    else:
        img_gray = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        img_gray = np.clip((img_gray * 255.0).round(), 0, 255) / 255.
        noise_gray = np.random.poisson(img_gray * vals).astype(np.float32) / vals - img_gray
        image += noise_gray[:, :, np.newaxis]
    image = np.clip(image, 0.0, 1.0)

    return image


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _add_speckle_noise(image: ndarray, noise_level1: int = 2, noise_level2: int = 25):
    noise_level = random.randint(noise_level1, noise_level2)
    image = np.clip(image, 0.0, 1.0)
    rnum = random.random()
    if rnum > 0.6:
        image += image * np.random.normal(0, noise_level / 255.0, image.shape).astype(np.float32)
    elif rnum < 0.4:
        image += image * np.random.normal(0, noise_level / 255.0, (*image.shape[:2], 1)).astype(np.float32)
    else:
        L = noise_level2 / 255.
        D = np.diag(np.random.rand(3))
        U = orth(np.random.rand(3, 3))
        conv = np.dot(np.dot(np.transpose(U), D), U)
        image += image * np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), image.shape[:2]).astype(
            np.float32)
    image = np.clip(image, 0.0, 1.0)

    return image


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _add_jpeg_compression(image: ndarray) -> ndarray:
    quality_factor = random.randint(30, 95)
    image = np.uint8((image.clip(0, 1) * 255.).round())
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encode_image = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
    image = cv2.imdecode(encode_image, 1)
    image = np.float32(image) / 255.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def _usm_sharp(image: np.ndarray, weight: float = 0.5, radius: int = 50, threshold: int = 10) -> np.ndarray:
    if radius % 2 == 0:
        radius += 1

    blur = cv2.GaussianBlur(image, (radius, radius), 0)
    residual = image - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype("float32")
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    out = image + weight * residual
    out = np.clip(out, 0, 1)
    out = soft_mask * out + (1 - soft_mask) * image

    return out


# Copy from `https://github.com/cszn/KAIR/blob/master/utils/utils_blindsr.py`
def _add_resize(image: ndarray, upscale_factor: int):
    image_height, image_width = image.shape[:2]
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.5 / upscale_factor, 1)
    else:
        sf1 = 1.0
    image = cv2.resize(image,
                       (int(sf1 * image_width), int(sf1 * image_height)),
                       interpolation=random.choice([1, 2, 3]))
    image = np.clip(image, 0.0, 1.0)

    return image


def image_to_tensor(image: ndarray, range_norm: bool, half: bool) -> Tensor:
    """Convert the image data type to the Tensor (NCWH) data type supported by PyTorch

    Args:
        image (np.ndarray): The image data read by ``OpenCV.imread``, the data range is [0,255] or [0, 1]
        range_norm (bool): Scale [0, 1] data to between [-1, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type

    Returns:
        tensor (Tensor): Data types supported by PyTorch

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=True, half=False)

    """
    # Convert image data type to Tensor data type
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float()

    # Scale the image data from [0, 1] to [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)

    # Convert torch.float32 image data type to torch.half image data type
    if half:
        tensor = tensor.half()

    return tensor


def tensor_to_image(tensor: Tensor, range_norm: bool, half: bool) -> Any:
    """Convert the Tensor(NCWH) data type supported by PyTorch to the np.ndarray(WHC) image data type

    Args:
        tensor (Tensor): Data types supported by PyTorch (NCHW), the data range is [0, 1]
        range_norm (bool): Scale [-1, 1] data to between [0, 1]
        half (bool): Whether to convert torch.float32 similarly to torch.half type.

    Returns:
        image (np.ndarray): Data types supported by PIL or OpenCV

    Examples:
        >>> example_image = cv2.imread("lr_image.bmp")
        >>> example_tensor = image_to_tensor(example_image, range_norm=False, half=False)

    """
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()

    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")

    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def image_resize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
    """Implementation of `imresize` function in Matlab under Python language.

    Args:
        image: The input image.
        scale_factor (float): Scale factor. The same scale applies for both height and width.
        antialiasing (bool): Whether to apply antialiasing when down-sampling operations.
            Caution: Bicubic down-sampling in `PIL` uses antialiasing by default. Default: ``True``.

    Returns:
        out_2 (np.ndarray): Output image with shape (c, h, w), [0, 1] range, w/o round

    """
    squeeze_flag = False
    if type(image).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(in_h, out_h, scale_factor, kernel_width,
                                                                              antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(in_w, out_w, scale_factor, kernel_width,
                                                                              antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(image)

    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2


def preprocess_one_image(image_path: str, device: torch.device) -> Tensor:
    image = cv2.imread(image_path).astype(np.float32) / 255.0

    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image data to pytorch format data
    tensor = image_to_tensor(image, False, False).unsqueeze_(0)

    # Transfer tensor channel image format data to CUDA device
    tensor = tensor.to(device=device, memory_format=torch.channels_last, non_blocking=True)

    return tensor


def degradation_process(
        image: ndarray,
        upscale_factor: int,
        jpeg_prob: float = 0.9,
        scale2_prob: float = 0.25,
) -> ndarray:
    """More detail see "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"""
    image_height, image_width = image.shape[:2]
    current_image_height, current_image_width = image.shape[:2]

    # First down-sample
    if upscale_factor == 4 and random.random() < scale2_prob:
        if np.random.rand() < 0.5:
            image = cv2.resize(image,
                               (int(1 / 2 * image_width), int(1 / 2 * image_height)),
                               interpolation=random.choice([1, 2, 3]))
        else:
            image = image_resize(image, 1 / 2, True)
        image = np.clip(image, 0.0, 1.0)
        upscale_factor = 2

    shuffle_order = random.sample(range(6), 6)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    # Keep three down-sample last
    if idx1 > idx2:
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:
        if i == 0:
            image = _add_blur(image, upscale_factor)
        elif i == 1:
            image = _add_blur(image, upscale_factor)
        elif i == 2:
            current_image_height, current_image_width = image.shape[:2]
            # Two down-sample
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * upscale_factor)
                image = cv2.resize(image,
                                   (int(1 / sf1 * image_width), int(1 / sf1 * image_height)),
                                   interpolation=random.choice([1, 2, 3]))
            else:
                k = _fspecial_gaussian(25, random.uniform(0.1, 0.6 * upscale_factor))
                k_shifted = _shift_pixel(k, upscale_factor)
                k_shifted = k_shifted / k_shifted.sum()  # blur with shifted kernel
                image = ndimage.filters.convolve(image, np.expand_dims(k_shifted, axis=2), mode="mirror")
                image = image[0::upscale_factor, 0::upscale_factor, ...]
            image = np.clip(image, 0.0, 1.0)
        elif i == 3:
            # Three down-sample
            image = cv2.resize(image,
                               (int(1 / upscale_factor * current_image_width),
                                int(1 / upscale_factor * current_image_height)),
                               interpolation=random.choice([1, 2, 3]))
            image = np.clip(image, 0.0, 1.0)
        elif i == 4:
            # Add Gaussian noise
            image = _add_gaussian_noise(image, noise_level1=2, noise_level2=25)
        elif i == 5:
            # Add JPEG noise
            if random.random() < jpeg_prob:
                image = _add_jpeg_compression(image)

    # Add final JPEG compression noise
    image = _add_jpeg_compression(image)

    return image


def degradation_process_plus(
        image: ndarray,
        upscale_factor: int,
        use_sharp: bool = True,
        shuffle_prob: float = 0.5,
        poisson_prob: float = 0.5,
        speckle_prob: float = 0.5,
) -> ndarray:
    """More detail see "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"""
    image_height, image_width = image.shape[:2]

    if use_sharp:
        image = _usm_sharp(image, 0.5, 50, 10)

    if random.random() < shuffle_prob:
        shuffle_order = random.sample(range(11), 11)
    else:
        shuffle_order = list(range(11))
        # local shuffle for noise, JPEG is always the last one
        shuffle_order[2:5] = random.sample(shuffle_order[2:5], len(range(2, 5)))
        shuffle_order[7:11] = random.sample(shuffle_order[7:11], len(range(7, 11)))

    for i in shuffle_order:
        if i == 0:
            image = _add_blur(image, upscale_factor)
        elif i == 1:
            image = _add_resize(image, upscale_factor)
        elif i == 2:
            image = _add_gaussian_noise(image, 2, 25)
        elif i == 3:
            if random.random() < poisson_prob:
                image = _add_poisson_noise(image)
        elif i == 4:
            if random.random() < speckle_prob:
                image = _add_speckle_noise(image)
        elif i == 5:
            image = _add_jpeg_compression(image)
        elif i == 6:
            image = _add_blur(image, upscale_factor)
        elif i == 7:
            image = _add_resize(image, upscale_factor)
        elif i == 8:
            image = _add_gaussian_noise(image, 2, 25)
        elif i == 9:
            if random.random() < poisson_prob:
                image = _add_poisson_noise(image)
        elif i == 10:
            if random.random() < speckle_prob:
                image = _add_speckle_noise(image)

    # resize to desired size
    image = cv2.resize(image,
                       (image_width // upscale_factor, image_height // upscale_factor),
                       interpolation=random.choice([1, 2, 3]))

    # Add final JPEG compression noise
    image = _add_jpeg_compression(image)

    return image


def expand_y(image: np.ndarray) -> np.ndarray:
    """Convert BGR channel to YCbCr format,
    and expand Y channel data in YCbCr, from HW to HWC

    Args:
        image (np.ndarray): Y channel image data

    Returns:
        y_image (np.ndarray): Y-channel image data in HWC form

    """
    # Normalize image data to [0, 1]
    image = image.astype(np.float32) / 255.

    # Convert BGR to YCbCr, and extract only Y channel
    y_image = bgr_to_ycbcr(image, only_use_y_channel=True)

    # Expand Y channel
    y_image = y_image[..., None]

    # Normalize the image data to [0, 255]
    y_image = y_image.astype(np.float64) * 255.0

    return y_image


def rgb_to_ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of rgb2ycbcr function in Matlab under Python language

    Args:
        image (np.ndarray): Image input in RGB format.
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        image = np.matmul(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


def bgr_to_ycbcr(image: np.ndarray, only_use_y_channel: bool) -> np.ndarray:
    """Implementation of bgr2ycbcr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in BGR format
        only_use_y_channel (bool): Extract Y channel separately

    Returns:
        image (np.ndarray): YCbCr image array data

    """
    if only_use_y_channel:
        image = np.dot(image, [24.966, 128.553, 65.481]) + 16.0
    else:
        image = np.matmul(image, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [
            16, 128, 128]

    image /= 255.
    image = image.astype(np.float32)

    return image


def ycbcr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2rgb function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): RGB image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0, -0.00153632, 0.00791071],
                              [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def ycbcr_to_bgr(image: np.ndarray) -> np.ndarray:
    """Implementation of ycbcr2bgr function in Matlab under Python language.

    Args:
        image (np.ndarray): Image input in YCbCr format.

    Returns:
        image (np.ndarray): BGR image array data

    """
    image_dtype = image.dtype
    image *= 255.

    image = np.matmul(image, [[0.00456621, 0.00456621, 0.00456621],
                              [0.00791071, -0.00153632, 0],
                              [0, -0.00318811, 0.00625893]]) * 255.0 + [-276.836, 135.576, -222.921]

    image /= 255.
    image = image.astype(image_dtype)

    return image


def rgb_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool) -> Tensor:
    """Implementation of rgb2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = Tensor([[65.481], [128.553], [24.966]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = Tensor([[65.481, -37.797, 112.0],
                         [128.553, -74.203, -93.786],
                         [24.966, 112.0, -18.214]]).to(tensor)
        bias = Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def bgr_to_ycbcr_torch(tensor: Tensor, only_use_y_channel: bool) -> Tensor:
    """Implementation of bgr2ycbcr function in Matlab under PyTorch

    References from：`https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion`

    Args:
        tensor (Tensor): Image data in PyTorch format
        only_use_y_channel (bool): Extract only Y channel

    Returns:
        tensor (Tensor): YCbCr image data in PyTorch format

    """
    if only_use_y_channel:
        weight = Tensor([[24.966], [128.553], [65.481]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = Tensor([[24.966, 112.0, -18.214],
                         [128.553, -74.203, -93.786],
                         [65.481, -37.797, 112.0]]).to(tensor)
        bias = Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    tensor /= 255.

    return tensor


def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image center area.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = (image_height - image_size) // 2
    left = (image_width - image_size) // 2

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_crop_np(image: np.ndarray, image_size: int) -> np.ndarray:
    """Crop small image patches from one image.

    Args:
        image (np.ndarray): The input image for `OpenCV.imread`.
        image_size (int): The size of the captured image area.

    Returns:
        patch_image (np.ndarray): Small patch image

    """
    image_height, image_width = image.shape[:2]

    # Just need to find the top and left coordinates of the image
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)

    # Crop image patch
    patch_image = image[top:top + image_size, left:left + image_size, ...]

    return patch_image


def random_crop(gt_tensor: Tensor,
                lr_tensor: Tensor,
                gt_image_size: int,
                upscale_factor: int) -> [Tensor, Tensor]:
    """Crop small image patches from one image.

    Args:
        gt_tensor (Tensor): High resolution images
        lr_tensor (Tensor): Low resolution images
        gt_image_size (int): The size of the captured high-resolution image area.
        upscale_factor (int): How many times the high-resolution image should be the low-resolution image

    Returns:
        patch_gt_tensor, patch_lr_tensor(Tensor, Tensor): Small gt patch images, small lr patch images

    """
    gt_image_height, hr_image_width = gt_tensor[0].size()[1:]

    # Just need to find the top and left coordinates of the image
    gt_top = random.randint(0, gt_image_height - gt_image_size)
    gt_left = random.randint(0, hr_image_width - gt_image_size)

    # Define the LR image position
    lr_top = gt_top // upscale_factor
    lr_left = gt_left // upscale_factor
    lr_image_size = gt_image_size // upscale_factor

    # Create patch images
    patch_gt_tensor = torch.zeros([gt_tensor.shape[0], gt_tensor.shape[1], gt_image_size, gt_image_size],
                                  dtype=lr_tensor.dtype,
                                  device=gt_tensor.device)
    patch_lr_tensor = torch.zeros([lr_tensor.shape[0], lr_tensor.shape[1], lr_image_size, lr_image_size],
                                  dtype=lr_tensor.dtype,
                                  device=lr_tensor.device)

    # Crop image patch
    for i in range(lr_tensor.shape[0]):
        patch_gt_tensor[i, :, :, :] = gt_tensor[i, :, gt_top:gt_top + gt_image_size, gt_left:gt_left + gt_image_size]
        patch_lr_tensor[i, :, :, :] = lr_tensor[i, :, lr_top:lr_top + lr_image_size, lr_left:lr_left + lr_image_size]

    return patch_gt_tensor, patch_lr_tensor


def random_rotate(image,
                  angles: list,
                  center: tuple[int, int] = None,
                  scale_factor: float = 1.0) -> np.ndarray:
    """Rotate an image by a random angle

    Args:
        image (np.ndarray): Image read with OpenCV
        angles (list): Rotation angle range
        center (optional, tuple[int, int]): High resolution image selection center point. Default: ``None``
        scale_factor (optional, float): scaling factor. Default: 1.0

    Returns:
        rotated_image (np.ndarray): image after rotation

    """
    image_height, image_width = image.shape[:2]

    if center is None:
        center = (image_width // 2, image_height // 2)

    # Random select specific angle
    angle = random.choice(angles)
    matrix = cv2.getRotationMatrix2D(center, angle, scale_factor)
    rotated_image = cv2.warpAffine(image, matrix, (image_width, image_height))

    return rotated_image


def random_horizontally_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip the image upside down randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Horizontally flip probability. Default: 0.5

    Returns:
        horizontally_flip_image (np.ndarray): image after horizontally flip

    """
    if random.random() < p:
        horizontally_flip_image = cv2.flip(image, 1)
    else:
        horizontally_flip_image = image

    return horizontally_flip_image


def random_vertically_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    """Flip an image horizontally randomly

    Args:
        image (np.ndarray): Image read with OpenCV
        p (optional, float): Vertically flip probability. Default: 0.5

    Returns:
        vertically_flip_image (np.ndarray): image after vertically flip

    """
    if random.random() < p:
        vertically_flip_image = cv2.flip(image, 0)
    else:
        vertically_flip_image = image

    return vertically_flip_image
