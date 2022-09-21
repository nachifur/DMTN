import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from src.image_pool import ImagePool

from src.loss import AdversarialLoss, PerceptualLoss, StyleLoss


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


class Discriminator(BaseNetwork):
    def __init__(self, config, in_channels, init_weights=True):
        super(Discriminator, self).__init__()
        # config
        self.config = config
        self.use_sigmoid = self.config.GAN_LOSS != 'hinge'
        norm = self.config.DIS_NORM
        # network
        self.conv1 = conv2d_layer(
            in_channels, 64, kernel_size=4, stride=2, norm=norm)
        self.conv2 = conv2d_layer(64, 128, kernel_size=4, stride=2, norm=norm)
        self.conv3 = conv2d_layer(128, 256, kernel_size=4, stride=2, norm=norm)
        self.conv4 = conv2d_layer(256, 512, kernel_size=4, stride=1, norm=norm)
        self.conv5 = conv2d_layer(512, 1, kernel_size=4, stride=1, norm=norm)
        # loss
        self.add_module('adversarial_loss',
                        AdversarialLoss(type=self.config.GAN_LOSS))
        self.add_module('perceptual_loss',
                        PerceptualLoss())
        self.fake_pool = ImagePool(self.config.POOL_SIZE)
        # optimizer
        self.optimizer = optim.Adam(
            params=self.parameters(),
            lr=float(config.LR_D),
            betas=(config.BETA1, config.BETA2)
        )
        if init_weights:
            self.init_weights()

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]

    def cal_loss(self, images, outputs, GT, cat_img=True):
        # discriminator loss
        loss = []
        dis_loss = 0
        if cat_img:
            dis_input_real = torch.cat((images, GT), dim=1)
            dis_input_fake = torch.cat((images, F.interpolate(self.fake_pool.query(outputs.detach()), outputs.shape[2:], mode="bilinear",align_corners=True)), dim=1)
        else:
            dis_input_real = GT
            dis_input_fake = F.interpolate(self.fake_pool.query(outputs.detach()), outputs.shape[2:], mode="bilinear",align_corners=True)

        dis_real, dis_real_feat = self(dis_input_real)
        dis_fake, dis_fake_feat = self(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2
        loss.append(dis_loss)

        # generator adversarial loss
        if cat_img:
            gen_input_fake = torch.cat((images, outputs), dim=1)
        else:
            gen_input_fake = outputs
        gen_fake, gen_fake_feat = self(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)

        # generator perceptual loss
        gen_perceptual_loss = self.perceptual_loss(outputs, GT)

        loss.append(gen_gan_loss)
        loss.append(gen_perceptual_loss)
        return loss

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def conv2d_layer(in_channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, norm="batch", activation_fn="LeakyReLU", conv_mode="none", pad_mode="ReflectionPad2d"):
    """
    norm: batch, spectral, instance, spectral_instance, none

    activation_fn: Sigmoid, ReLU, LeakyReLU, none

    conv_mode: transpose, upsample, none

    pad_mode: ReflectionPad2d, ReplicationPad2d, ZeroPad2d
    """
    layer = []
    # padding
    if conv_mode == "transpose":
        pass
    else:
        if pad_mode == "ReflectionPad2d":
            layer.append(nn.ReflectionPad2d(padding))
        elif pad_mode == "ReplicationPad2d":
            layer.append(nn.ReflectionPad2d(padding))
        else:
            layer.append(nn.ZeroPad2d(padding))
        padding = 0

    # conv layer
    if norm == "spectral" or norm == "spectral_instance":
        bias = False
        # conv
        if conv_mode == "transpose":
            conv_ = nn.ConvTranspose2d
        elif conv_mode == "upsample":
            layer.append(nn.Upsample(mode='bilinear', scale_factor=stride))
            conv_ = nn.Conv2d
        else:
            conv_ = nn.Conv2d
    else:
        bias = True
        # conv
        if conv_mode == "transpose":
            layer.append(nn.ConvTranspose2d(in_channels, channels, kernel_size,
                                            bias=bias, stride=stride, padding=padding, dilation=dilation))
        elif conv_mode == "upsample":
            layer.append(nn.Upsample(mode='bilinear', scale_factor=stride))
            layer.append(nn.Conv2d(in_channels, channels, kernel_size,
                                   bias=bias, stride=stride, padding=padding, dilation=dilation))
        else:
            layer.append(nn.Conv2d(in_channels, channels, kernel_size,
                                   bias=bias, stride=stride, padding=padding, dilation=dilation))

    # norm
    if norm == "spectral":
        layer.append(spectral_norm(conv_(in_channels, channels, kernel_size,
                                         stride=stride, bias=bias, padding=padding, dilation=dilation), True))
    elif norm == "instance":
        layer.append(nn.InstanceNorm2d(
            channels, affine=True, track_running_stats=False))
    elif norm == "batch":
        layer.append(nn.BatchNorm2d(
            channels, affine=True, track_running_stats=False))
    elif norm == "spectral_instance":
        layer.append(spectral_norm(conv_(in_channels, channels, kernel_size,
                                         stride=stride, bias=bias, padding=padding, dilation=dilation), True))
        layer.append(nn.InstanceNorm2d(
            channels, affine=True, track_running_stats=False))
    elif norm == "batch_":
        layer.append(BatchNorm_(channels))
    else:
        pass

    # activation_fn
    if activation_fn == "Sigmoid":
        layer.append(nn.Sigmoid())
    elif activation_fn == "ReLU":
        layer.append(nn.ReLU(True))
    elif activation_fn == "none":
        pass
    else:
        layer.append(nn.LeakyReLU(0.2,inplace=True))

    return nn.Sequential(*layer)


class BatchNorm_(nn.Module):
    def __init__(self, channels):
        super(BatchNorm_, self).__init__()
        self.w0 = torch.nn.Parameter(
            torch.FloatTensor([1.0]), requires_grad=True)
        self.w1 = torch.nn.Parameter(
            torch.FloatTensor([0.0]), requires_grad=True)
        self.BatchNorm2d = nn.BatchNorm2d(
            channels, affine=True, track_running_stats=False)

    def forward(self, x):
        outputs = self.w0*x+self.w1*self.BatchNorm2d(x)
        return outputs

def avgcov2d_layer(pool_kernel_size, pool_stride, in_channels, channels, conv_kernel_size=3, conv_stride=1, padding=1, dilation=1, norm="batch", activation_fn="LeakyReLU"):
    layer = []
    layer.append(nn.AvgPool2d(pool_kernel_size, pool_stride))
    layer.append(conv2d_layer(in_channels, channels, kernel_size=conv_kernel_size, stride=conv_stride,
                              padding=padding, dilation=dilation, norm=norm, activation_fn=activation_fn))
    return nn.Sequential(*layer)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def get_encoder(encoder_param, norm):
    encoder = []
    index = 0
    for param in encoder_param:
        if index == 0:
            encoder.append(conv2d_layer(param[0], param[1], kernel_size=param[2], stride=param[3], padding=param[4], dilation=1, norm=norm,
                                        activation_fn="ReLU", conv_mode="none", pad_mode="ReflectionPad2d"))
        else:
            encoder.append(conv2d_layer(param[0], param[1], kernel_size=param[2], stride=param[3], padding=param[4], dilation=1, norm=norm,
                                        activation_fn="ReLU", conv_mode="none", pad_mode="ZeroPad2d"))
        index += 1
    return encoder


def get_middle(middle_param, norm):
    blocks = []
    for _ in range(middle_param[0]):
        block = ResnetBlock(
            middle_param[1], norm)
        blocks.append(block)
    return blocks


def get_decoder(decoder_param, norm, Sigmoid=True):
    if Sigmoid:
        activation_fn = "Sigmoid"
    else:
        activation_fn = "none"
    decoder = []
    index = 0
    for param in decoder_param:
        if index == len(decoder_param)-1:
            decoder.append(conv2d_layer(param[0], param[1], kernel_size=param[2], stride=param[3], padding=param[4], dilation=1, norm="none",
                                        activation_fn=activation_fn, conv_mode="none", pad_mode="ReflectionPad2d"))
        else:
            decoder.append(conv2d_layer(param[0], param[1], kernel_size=param[2], stride=param[3], padding=param[4], dilation=1, norm=norm,
                                        activation_fn="ReLU", conv_mode="transpose", pad_mode="ZeroPad2d"))
        index += 1
    return decoder


def get_encoder_decoder(in_channels, ResnetBlockNum, Sigmoid=True, norm="batch"):
    encoder_param = [
        [in_channels, 64, 7, 1, 3],
        [64, 128, 4, 2, 1],
        [128, 256, 4, 2, 1]]
    encoder = nn.Sequential(
        *get_encoder(encoder_param, norm))
    middle_param = [ResnetBlockNum, 256]
    middle = nn.Sequential(
        *get_middle(middle_param, norm))
    decoder_param = [
        [256, 128, 4, 2, 1],
        [128, 64, 4, 2, 1],
        [64, 3, 7, 1, 3]]
    decoder = nn.Sequential(
        *get_decoder(decoder_param, norm, Sigmoid=Sigmoid))
    return nn.Sequential(*[encoder, middle, decoder])


class ResnetBlock(nn.Module):
    def __init__(self, dim, norm):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            conv2d_layer(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, norm=norm,
                         activation_fn="ReLU", conv_mode="none", pad_mode="ReflectionPad2d"),

            conv2d_layer(dim, dim, kernel_size=3, stride=1, padding=2, dilation=2, norm=norm,
                         activation_fn="ReLU", conv_mode="none", pad_mode="ReflectionPad2d"),

            conv2d_layer(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, norm=norm,
                         activation_fn="ReLU", conv_mode="none", pad_mode="ReflectionPad2d"),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
