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
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
from torch.nn.utils import spectral_norm
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

__all__ = [
    "Discriminator", "BSRGAN", "ContentLoss",
    "discriminator", "lsrgan_x2", "lsrgan_x4", "content_loss", "content_loss_for_vgg19_34",

]


class _ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels + growth_channels * 0, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growth_channels * 1, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growth_channels * 2, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growth_channels * 3, growth_channels, (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class _ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(_ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = _ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = _ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))
        self.down_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 128, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 256, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.down_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 512, (4, 4), (2, 2), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(512, 256, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block2 = nn.Sequential(
            spectral_norm(nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.up_block3 = nn.Sequential(
            spectral_norm(nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False)),
            nn.LeakyReLU(0.2, True),
        )
        self.conv4 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)

        # Down-sampling
        down1 = self.down_block1(out1)
        down2 = self.down_block2(down1)
        down3 = self.down_block3(down2)

        # Up-sampling
        down3 = F.interpolate(down3, scale_factor=2, mode="bilinear", align_corners=False)
        up1 = self.up_block1(down3)

        up1 = torch.add(up1, down2)
        up1 = F.interpolate(up1, scale_factor=2, mode="bilinear", align_corners=False)
        up2 = self.up_block2(up1)

        up2 = torch.add(up2, down1)
        up2 = F.interpolate(up2, scale_factor=2, mode="bilinear", align_corners=False)
        up3 = self.up_block3(up2)

        up3 = torch.add(up3, out1)

        out = self.conv2(up3)
        out = self.conv3(out)
        out = self.conv4(out)

        return out


class BSRGAN(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            channels: int = 64,
            growth_channels: int = 32,
            num_blocks: int = 23,
            upscale_factor: int = 4,
    ) -> None:
        super(BSRGAN, self).__init__()
        self.upscale_factor = upscale_factor

        # The first layer of convolutional layer.
        self.conv1 = nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network.
        trunk = []
        for _ in range(num_blocks):
            trunk.append(_ResidualResidualDenseBlock(channels, growth_channels))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks.
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer.
        self.upsampling1 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        if upscale_factor == 4:
            self.upsampling2 = nn.Sequential(
                nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
                nn.LeakyReLU(0.2, True)
            )

        # Reconnect a layer of convolution block after upsampling.
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # Output layer.
        self.conv4 = nn.Conv2d(channels, out_channels, (3, 3), (1, 1), (1, 1))

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv1(x)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)

        out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
        if self.upscale_factor == 4:
            out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))

        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ContentLoss(nn.Module):
    """Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.

    Paper reference list:
        -`Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        -`ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        -`Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.

     """

    def __init__(
            self,
            feature_model_extractor_nodes: list,
            feature_model_normalize_mean: list,
            feature_model_normalize_std: list
    ) -> None:
        super(ContentLoss, self).__init__()
        # Get the name of the specified feature extraction node
        self.feature_model_extractor_nodes = feature_model_extractor_nodes
        # Load the VGG19 model trained on the ImageNet dataset.
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # Extract the thirty-fifth layer output in the VGG19 model as the content loss.
        self.feature_extractor = create_feature_extractor(model, feature_model_extractor_nodes)
        # set to validation mode
        self.feature_extractor.eval()

        # The preprocessing method of the input data.
        # This is the VGG model preprocessing method of the ImageNet dataset
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)

        # Freeze model parameters.
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> [torch.Tensor,
                                                                            torch.Tensor,
                                                                            torch.Tensor,
                                                                            torch.Tensor,
                                                                            torch.Tensor]:
        # Standardized operations
        sr_tensor = self.normalize(sr_tensor)
        gt_tensor = self.normalize(gt_tensor)

        sr_features = self.feature_extractor(sr_tensor)
        gt_features = self.feature_extractor(gt_tensor)

        # Find the feature map difference between the two images
        loss1 = F.l1_loss(sr_features[self.feature_model_extractor_nodes[0]],
                          gt_features[self.feature_model_extractor_nodes[0]])
        loss2 = F.l1_loss(sr_features[self.feature_model_extractor_nodes[1]],
                          gt_features[self.feature_model_extractor_nodes[1]])
        loss3 = F.l1_loss(sr_features[self.feature_model_extractor_nodes[2]],
                          gt_features[self.feature_model_extractor_nodes[2]])
        loss4 = F.l1_loss(sr_features[self.feature_model_extractor_nodes[3]],
                          gt_features[self.feature_model_extractor_nodes[3]])
        loss5 = F.l1_loss(sr_features[self.feature_model_extractor_nodes[4]],
                          gt_features[self.feature_model_extractor_nodes[4]])

        return loss1, loss2, loss3, loss4, loss5


def discriminator() -> Discriminator:
    model = Discriminator()

    return model


def bsrgan_x2(**kwargs: Any) -> BSRGAN:
    model = BSRGAN(upscale_factor=2, **kwargs)

    return model


def lsrgan_x4(**kwargs: Any) -> BSRGAN:
    model = BSRGAN(upscale_factor=4, **kwargs)

    return model


def content_loss(feature_model_extractor_nodes,
                 feature_model_normalize_mean,
                 feature_model_normalize_std) -> ContentLoss:
    content_loss = ContentLoss(feature_model_extractor_nodes,
                               feature_model_normalize_mean,
                               feature_model_normalize_std)

    return content_loss
