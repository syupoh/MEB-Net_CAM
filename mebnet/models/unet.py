import pdb
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import numpy as np

# from munch import Munch
import math
import torch.nn.functional as F

# class AdaIN(nn.Module):
#     def __init__(self, style_dim, num_features):
#         super().__init__()
#         self.norm = nn.InstanceNorm2d(num_features, affine=False)
#         self.fc = nn.Linear(style_dim, num_features*2)
#
#     def forward(self, x, s):
#         h = self.fc(s)
#         h = h.view(h.size(0), h.size(1), 1, 1)
#         gamma, beta = torch.chunk(h, chunks=2, dim=1)
#         return (1 + gamma) * self.norm(x) + beta
#
#
# class AdainResBlk(nn.Module):
#     def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
#                  actv=nn.LeakyReLU(0.2), upsample=False):
#         super().__init__()
#         self.w_hpf = w_hpf
#         self.actv = actv
#         self.upsample = upsample
#         self.learned_sc = dim_in != dim_out
#         self._build_weights(dim_in, dim_out, style_dim)
#
#     def _build_weights(self, dim_in, dim_out, style_dim=64):
#         self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
#         self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
#         self.norm1 = AdaIN(style_dim, dim_in)
#         self.norm2 = AdaIN(style_dim, dim_out)
#         if self.learned_sc:
#             self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
#
#     def _shortcut(self, x):
#         if self.upsample:
#             x = F.interpolate(x, scale_factor=2, mode='nearest')
#         if self.learned_sc:
#             x = self.conv1x1(x)
#         return x
#
#     def _residual(self, x, s):
#         x = self.norm1(x, s)
#         x = self.actv(x)
#         if self.upsample:
#             x = F.interpolate(x, scale_factor=2, mode='nearest')
#         x = self.conv1(x)
#         x = self.norm2(x, s)
#         x = self.actv(x)
#         x = self.conv2(x)
#         return x
#
#     def forward(self, x, s):
#         out = self._residual(x, s)
#         if self.w_hpf == 0:
#             out = (out + self._shortcut(x)) / math.sqrt(2)
#         return out
#
#
# class HighPass(nn.Module):
#     def __init__(self, w_hpf, device):
#         super(HighPass, self).__init__()
#         self.filter = torch.tensor([[-1, -1, -1],
#                                     [-1, 8., -1],
#                                     [-1, -1, -1]]).to(device) / w_hpf
#
#     def forward(self, x):
#         filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
#         return F.conv2d(x, filter, padding=1, groups=x.size(1))

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class WNetDownConvBlock(nn.Module):
    r"""Performs two 3x3 2D convolutions, each followed by a ReLU and batch norm. Ends with a 2D max-pool operation."""

    def __init__(self, in_features: int, out_features: int):
        r"""
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(WNetDownConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, out_features, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_features),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_features, out_features, 3),
            nn.ReLU(),
            nn.BatchNorm2d(out_features),
            nn.ReplicationPad2d(1),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        feature_map = self.layers(x)
        return self.pool(feature_map), feature_map


class WNetUpConvBlock(nn.Module):
    r"""Performs two 3x3 2D convolutions, each followed by a ReLU and batch norm. Ends with a transposed convolution with a stride of 2 on the last layer. Halves features at first and third convolutions"""

    def __init__(self, in_features: int, mid_features: int, out_features: int):
        r"""
        :param in_features: Number of feature channels in the incoming data
        :param out_features: Number of feature channels in the outgoing data
        """
        super(WNetUpConvBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 3),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid_features, mid_features, 3),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.ConvTranspose2d(mid_features, out_features, 2, stride=2),
        )


    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)


class WNetOutputBlock(nn.Module):
    r"""Performs two 3x3 2D convolutions, each followed by a ReLU and batch Norm.
    Ending with a 1x1 convolution to map features to classes."""

    def __init__(self, in_features: int, num_classes: int):
        r"""
        :param in_features: Number of feature channels in the incoming data
        :param num_classes: Number of feature channels in the outgoing data
        """
        super(WNetOutputBlock, self).__init__()
        mid_features = int(in_features / 2)
        self.layers = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 3),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid_features, mid_features, 3),
            nn.BatchNorm2d(mid_features),
            nn.ReLU(),
            nn.ReplicationPad2d(1),

            # 1x1 convolution to map features to classes
            nn.Conv2d(mid_features, num_classes, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Module outputs
        """
        return self.layers(x)

#### with unet_discriminator
# class UNetAuto(nn.Module):
#     r"""UNet based architecture for image auto encoding"""
#
#     def __init__(self, num_channels: int = 3, num_out_channels: int = 3,
#                  max_features: int = 1024, batch_size: int = 16):
#         r"""
#         :param num_channels: Number of channels in the raw image data
#         :param num_out_channels: Number of channels in the output data
#         """
#         super(UNetAuto, self).__init__()
#
#         self.batch_size = batch_size
#         if max_features not in [1024, 512, 256]:
#             print('Max features restricted to [1024, 512, 256]')
#             max_features = 1024
#         features_4 = max_features // 2
#         features_3 = features_4 // 2
#         features_2 = features_3 // 2
#         features_1 = features_2 // 2
#
#         self.conv_block1 = WNetDownConvBlock(num_channels, features_1)
#         self.conv_block2 = WNetDownConvBlock(features_1, features_2)
#         self.conv_block3 = WNetDownConvBlock(features_2, features_3)
#         self.conv_block4 = WNetDownConvBlock(features_3, features_4)
#
#         self.d = unet_Discriminator(features_4 * 16 * 8)
#
#         self.deconv_block1 = WNetUpConvBlock(features_4, max_features, features_4)
#         self.deconv_block2 = WNetUpConvBlock(max_features, features_4, features_3)
#         self.deconv_block3 = WNetUpConvBlock(features_4, features_3, features_2)
#         self.deconv_block4 = WNetUpConvBlock(features_3, features_2, features_1)
#
#         self.output_block = WNetOutputBlock(features_2, num_out_channels)
#
#
#     def forward(self, x: Tensor, ) -> Tensor:
#         """Pushes a set of inputs (x) through the network.
#
#         :param x: Input values
#         :return: Network output Tensor
#         """
#
#         # print(f'Block: 0 Curr shape: {x.shape}')
#         x, c1 = self.conv_block1(x)
#         # print(f'Block: 1 Out shape: {x.shape}; features shape: {c1.shape}')
#         x, c2 = self.conv_block2(x)
#         # print(f'Block: 2 Out shape: {x.shape}; features shape: {c2.shape}')
#         x, c3 = self.conv_block3(x)
#         # print(f'Block: 3 Out shape: {x.shape}; features shape: {c3.shape}')
#         x, c4 = self.conv_block4(x)
#         # print(f'Block: 4 Out shape: {x.shape}; features shape: {c4.shape}')
#
#         validity = x.view([self.batch_size, -1])
#         # validity = self.d(x.view([self.batch_size, -1]))
#
#         # print(x.shape) # c x h x w => features4, h/16, w/16
#
#         d1 = self.deconv_block1(x)
#         # print(f'Block: 5 Out shape: {d1.shape}')
#         d2 = self.deconv_block2(torch.cat((c4, d1), dim=1))
#         # print(f'Block: 6 Out shape: {d2.shape}')
#         d3 = self.deconv_block3(torch.cat((c3, d2), dim=1))
#         # print(f'Block: 7 Out shape: {d3.shape}')
#         d4 = self.deconv_block4(torch.cat((c2, d3), dim=1))
#         # print(f'Block: 8 Out shape: {d4.shape}')
#         out = self.output_block(torch.cat((c1, d4), dim=1))
#         # print(f'Block: 9 Out shape: {out.shape}')
#
#         return out, validity

class UNet3Step(nn.Module):
    r"""Smaller UNet based architecture for image auto encoding"""

    def __init__(self, num_channels: int = 3, num_out_channels: int = 3, max_features: int = 1024):
        r"""
        :param num_channels: Number of channels in the raw image data
        :param num_out_channels: Number of channels in the output data
        """
        super(UNet3Step, self).__init__()
        if max_features not in [1024, 512, 256, 128]:
            print('Max features restricted to [1024, 512, 256, 128]')
            max_features = 1024
        features_3 = max_features // 2
        features_2 = features_3 // 2
        features_1 = features_2 // 2

        self.conv_block1 = WNetDownConvBlock(num_channels, features_1)
        self.conv_block2 = WNetDownConvBlock(features_1, features_2)
        self.conv_block3 = WNetDownConvBlock(features_2, features_3)

        self.deconv_block1 = WNetUpConvBlock(features_3, max_features, features_3)
        self.deconv_block2 = WNetUpConvBlock(max_features, features_3, features_2)
        self.deconv_block3 = WNetUpConvBlock(features_3, features_2, features_1)

        self.output_block = WNetOutputBlock(features_2, num_out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network output Tensor
        """
        # print(f'Block: 0 Curr shape: {x.shape}')
        x, c1 = self.conv_block1(x)
        # print(f'Block: 1 Out shape: {x.shape}; features shape: {c1.shape}')
        x, c2 = self.conv_block2(x)
        # print(f'Block: 2 Out shape: {x.shape}; features shape: {c2.shape}')
        x, c3 = self.conv_block3(x)
        # print(f'Block: 3 Out shape: {x.shape}; features shape: {c3.shape}')
        d1 = self.deconv_block1(x)
        # print(f'Block: 4 Out shape: {d1.shape}')
        d2 = self.deconv_block2(torch.cat((c3, d1), dim=1))
        # print(f'Block: 5 Out shape: {d2.shape}')
        d3 = self.deconv_block3(torch.cat((c2, d2), dim=1))
        # print(f'Block: 6 Out shape: {d3.shape}')
        out = self.output_block(torch.cat((c1, d3), dim=1))
        # print(f'Block: 7 Out shape: {out.shape}')

        return out


class UNetAuto(nn.Module):
    r"""UNet based architecture for image auto encoding"""

    def __init__(self, num_channels: int = 3, num_out_channels: int = 3,
                 max_features: int = 1024, batch_size: int = 16):
        r"""
        :param num_channels: Number of channels in the raw image data
        :param num_out_channels: Number of channels in the output data
        """
        super(UNetAuto, self).__init__()

        self.batch_size = batch_size
        if max_features not in [1024, 512, 256]:
            print('Max features restricted to [1024, 512, 256]')
            max_features = 1024
        features_4 = max_features // 2
        features_3 = features_4 // 2
        features_2 = features_3 // 2
        features_1 = features_2 // 2

        self.conv_block1 = WNetDownConvBlock(num_channels, features_1)
        self.conv_block2 = WNetDownConvBlock(features_1, features_2)
        self.conv_block3 = WNetDownConvBlock(features_2, features_3)
        self.conv_block4 = WNetDownConvBlock(features_3, features_4)

        self.deconv_block1 = WNetUpConvBlock(features_4, max_features, features_4)
        self.deconv_block2 = WNetUpConvBlock(max_features, features_4, features_3)
        self.deconv_block3 = WNetUpConvBlock(features_4, features_3, features_2)
        self.deconv_block4 = WNetUpConvBlock(features_3, features_2, features_1)

        self.output_block = WNetOutputBlock(features_2, num_out_channels)


    def forward(self, x: Tensor, ) -> Tensor:
        """Pushes a set of inputs (x) through the network.

        :param x: Input values
        :return: Network output Tensor
        """

        # print(f'Block: 0 Curr shape: {x.shape}')
        x, c1 = self.conv_block1(x)
        # print(f'Block: 1 Out shape: {x.shape}; features shape: {c1.shape}')
        x, c2 = self.conv_block2(x)
        # print(f'Block: 2 Out shape: {x.shape}; features shape: {c2.shape}')
        x, c3 = self.conv_block3(x)
        # print(f'Block: 3 Out shape: {x.shape}; features shape: {c3.shape}')
        x, c4 = self.conv_block4(x)
        # print(f'Block: 4 Out shape: {x.shape}; features shape: {c4.shape}')

        # print(x.shape) # c x h x w => features4, h/16, w/16

        d1 = self.deconv_block1(x)
        # print(f'Block: 5 Out shape: {d1.shape}')
        d2 = self.deconv_block2(torch.cat((c4, d1), dim=1))
        # print(f'Block: 6 Out shape: {d2.shape}')
        d3 = self.deconv_block3(torch.cat((c3, d2), dim=1))
        # print(f'Block: 7 Out shape: {d3.shape}')
        d4 = self.deconv_block4(torch.cat((c2, d3), dim=1))
        # print(f'Block: 8 Out shape: {d4.shape}')
        out = self.output_block(torch.cat((c1, d4), dim=1))
        # print(f'Block: 9 Out shape: {out.shape}')

        return out

# class unet_Discriminator(nn.Module):
#     def __init__(self, latent_dim):
#         super(Discriminator, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(latent_dim, 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, z):
#
#         validity = self.model(z)
#         return validity


# class Discriminator(nn.Module): # starGAN
#     def __init__(self, img_shape=(3, 256, 128), c_dim=5, n_strided=6):
#         super(Discriminator, self).__init__()
#         channels, height, width = img_shape
#         self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)
#
#         # print(img_size)
#         def discriminator_block(in_filters, out_filters):
#             """Returns downsampling layers of each discriminator block"""
#             layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1), nn.LeakyReLU(0.01)]
#             return layers
#
#         layers = discriminator_block(channels, 64)
#         curr_dim = 64
#         for _ in range(n_strided - 1):
#             layers.extend(discriminator_block(curr_dim, curr_dim * 2))
#             curr_dim *= 2
#
#         self.model = nn.Sequential(*layers)
#
#         # Output 1: PatchGAN
#         self.out1 = nn.Conv2d(curr_dim, 1, 3, padding=1, bias=False)
#         # Output 2: Class prediction
#         kernel_size = width // 2 ** n_strided
#         self.out2 = nn.Conv2d(curr_dim, c_dim, kernel_size, bias=False)
#
#     def forward(self, img):
#         feature_repr = self.model(img)
#         out_adv = self.out1(feature_repr)
#         out_cls = self.out2(feature_repr)
#         return out_adv, out_cls.view(out_cls.size(0), -1)

class Discriminator(nn.Module): # CycleGAN
    def __init__(self, input_shape=(3, 256, 128)):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape

        # Calculate output shape of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 4, width // 2 ** 4)

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img):
        return self.model(img)

class Discriminator_st2(nn.Module): # StarGANv2
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv2d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, dim_out, 2, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y=0):
        out = self.main(x) # [1, 2, 1, 1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out



def get_model(string_name):
    if string_name in ['unet256', 'unet512', 'unet1024']:
        features = int(string_name[4:])
    elif string_name in ['smallu128', 'smallu256', 'smallu512', 'smallu1024']:
        features = int(string_name[6:])
        return UNet3Step(max_features=features)
    elif string_name in ['discriminator']:
        return Discriminator()
    elif string_name in ['Discriminator_st2']:
        return Discriminator_st2()
    else:
        features = 512
    return UNetAuto(max_features=features)


if __name__ == "__main__":
    # models = ['unet256', 'unet512', 'unet1024', 'smallu128', 'smallu256', 'smallu512', 'smallu1024']
    # models = ['unet1024']
    models = ['discriminator', 'Discriminator_st2']

    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ##############
    ##############

    #
    # pdb.set_trace()
    ##############
    ##############

    for name in models:
        model = get_model(name)
        print(f'Model: {name}')
        print(model)
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        #     model.to('cuda')
        try:
            print(model)
            from torchsummary import summary
            print('== Start ==')
            summary(model, input_size=(3, 256, 128))
            print('== End ==')
        except:
            if name=='Discriminator_st2':
                from torch.autograd import Variable
                batchnum = 3
                Tensor = torch.cuda.FloatTensor

                fake = Variable(Tensor(np.ones((batchnum, 3, 256, 128))), requires_grad=False)
                fake_y = torch.LongTensor((np.random.randint(2, size=(batchnum))))
                output = model(fake, fake_y)
                output
            else:
                print(model)
                print('torchsummary failed: pip install torchsummary or bad input')
    ####
        # Tensor = torch.cuda.FloatTensor
        # from torch.autograd import Variable
        # example_input = Variable(Tensor(1, 3, 256, 128).fill_(1.0), requires_grad=False)
        # output = model(example_input)
        # print(output[0].shape)
        # print(output[1].shape)
        # print(output[0])
        # print(output[1])`
    ####
        del model
    print(f"tested {len(models)} model types: {models}")

    # output[1].shape
    # torch.Size([1, 15])
    #
    # output[0].shape
    # torch.Size([1, 1, 4, 2])



    # """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # model()
