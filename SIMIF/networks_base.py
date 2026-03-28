from collections import OrderedDict

import torch
import torch.nn as nn


def weights_init(m):
    """
    Custom weight initialization function.

    Args:
        m (torch.nn.Module): Module to initialize.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
def initialize_weights(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:  # 检查 m.bias 是否为 None
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, 0, 0.01)
        if module.bias is not None:  # 检查 m.bias 是否为 None
            nn.init.constant_(module.bias, 0)


class Generator(nn.Module):
    """
    Generator network for the CGAN model.

    Args:
        in_channels (int): Number of input channels (default: 3).
        out_channels (int): Number of output channels (default: 1).
        init_features (int): Number of initial features (default: 32).

    References:
    - Ronneberger, O., Fischer, P. and Brox, T., 2015.
      U-net: Convolutional networks for biomedical image segmentation.
      In Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015:
      18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III (pp. 234-241).
      Springer International Publishing.
    """

    def __init__(self, in_channels=3, out_channels=1, init_features=128):
        super(Generator, self).__init__()

        features = init_features
        self.encoder1 = Generator._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Generator._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Generator._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Generator._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Generator._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Generator._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Generator._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Generator._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Generator._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        """
        Forward pass of the generator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Generated output tensor.

        References:
        - Isola, P., Zhu, J.Y., Zhou, T. and Efros, A.A., 2017.
         Image-to-image translation with conditional adversarial networks.
         In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        dec1 = torch.sigmoid(self.conv(dec1))
        return dec1

    @staticmethod
    def _block(in_channels, features, name):
        """
        Helper function to create a block of convolutional layers with batch normalization and ReLU activation.

        Args:
            in_channels (int): Number of input channels.
            features (int): Number of output channels.
            name (str): Name prefix for the layers.

        Returns:
            torch.nn.Sequential: Block of convolutional layers.
        """
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class CNNBlock(nn.Module):
    """
    Convolutional block used in the discriminator.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride value for the convolutional layers (default: 2).
    """

    def __init__(self, in_channels, out_channels, stride=2):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        """
        Forward pass of the convolutional block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, real_channels=1, gen_channels=1):
        super(Discriminator, self).__init__()

        act = nn.LeakyReLU(0.2, inplace=True)
        input_channel = real_channels+gen_channels
        n_df = 64
        norm = nn.InstanceNorm2d

        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=1), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        #x = torch.concat([x,y],dim=1)
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            x = block(x)
        #x = torch.sigmoid(x)
        return x
