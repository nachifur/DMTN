import numpy as np
from numpy.core.fromnumeric import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.loss import VGG19
from src.network.networks import Discriminator, avgcov2d_layer, conv2d_layer
from torch.autograd import Variable
from src.utils import solve_factor, imshow
import torchvision.transforms.functional as TF
import scipy.linalg


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def init_weights(self, init_type='identity', gain=0.02):
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
                    nn.init.kaiming_normal_(
                        m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == "identity":
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, gain)
                    else:
                        identity_initializer(m.weight.data)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


def identity_initializer(data):
    shape = data.shape
    array = np.zeros(shape, dtype=float)
    cx, cy = shape[2]//2, shape[3]//2
    for i in range(np.minimum(shape[0], shape[1])):
        array[i, i, cx, cy] = 1
    return torch.tensor(array, dtype=torch.float32)


class DMTN(BaseNetwork):
    """DMTN"""

    def __init__(self, config, in_channels=3, init_weights=True):
        super(DMTN, self).__init__()

        # gan
        channels = 64
        stage_num = [12, 2]
        self.network = DMTNSOURCE(
            in_channels, channels, norm=config.GAN_NORM, stage_num=stage_num)
        # gan loss
        if config.LOSS == "MSELoss":
            self.add_module('loss', nn.MSELoss(reduction="mean"))
        elif config.LOSS == "L1Loss":
            self.add_module('loss', nn.L1Loss(reduction="mean"))

        # gan optimizer
        self.optimizer = optim.Adam(
            params=self.network.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

        # dis
        self.ADV = config.ADV
        if self.ADV:
            discriminator = []
            discriminator.append(Discriminator(config, in_channels=6))
            self.discriminator = nn.Sequential(*discriminator)

        if init_weights:
            self.init_weights(config.INIT_TYPE)

    def process(self, images, mask, GT):
        loss = []
        logs = []
        inputs = images
        img, net_matte, outputs = self(
            inputs)

        matte_gt = GT-inputs
        matte_gt = matte_gt - \
            (matte_gt.min(dim=2, keepdim=True).values).min(
                dim=3, keepdim=True).values
        matte_gt = matte_gt / \
            (matte_gt.max(dim=2, keepdim=True).values).max(
                dim=3, keepdim=True).values
        match_loss_1 = 0
        match_loss_2 = 0
        matte_loss = 0

        match_loss_1 += self.cal_loss(outputs, GT)*255*10
        match_loss_2 += self.cal_loss(img, images)*255*10
        matte_loss += self.cal_loss(net_matte, matte_gt)*255

        if self.ADV:
            dis_loss_1, gen_gan_loss_1, perceptual_loss_1 = self.discriminator[0].cal_loss(
                images, outputs, GT)

            perceptual_loss_1 = perceptual_loss_1*1000

            gen_loss = perceptual_loss_1 + match_loss_1+match_loss_2 +\
                matte_loss+gen_gan_loss_1

            loss.append(gen_loss)
            loss.append(dis_loss_1)

            logs.append(("l_match1", match_loss_1.item()))
            logs.append(("l_match2", match_loss_2.item()))
            logs.append(("l_matte", matte_loss.item()))
            logs.append(("l_perceptual_1", perceptual_loss_1.item()))
            logs.append(("l_adv1", gen_gan_loss_1.item()))
            logs.append(("l_gen", gen_loss.item()))
            logs.append(("l_dis1", dis_loss_1.item()))
        else:
            gen_loss = match_loss_1 + match_loss_2 + matte_loss
            gen_loss = gen_loss
            loss.append(gen_loss)

            logs.append(("l_match1", match_loss_1.item()))
            logs.append(("l_match2", match_loss_2.item()))
            logs.append(("l_matte", matte_loss.item()))
            logs.append(("l_gen", gen_loss.item()))

        return [net_matte, outputs], loss, logs

    def forward(self, x):
        outputs = self.network(x)
        return outputs

    def cal_loss(self, outputs, GT):
        matching_loss = self.loss(outputs, GT)
        return matching_loss

    def backward(self, loss):
        self.optimizer.zero_grad()
        loss[0].backward()
        self.optimizer.step()
        if self.ADV:
            i = 0
            for discriminator in self.discriminator:
                discriminator.backward(loss[1+i])
                i += 1


class FeatureDecouplingModule(nn.Module):
    '''Shadow feature decoupling module'''

    def __init__(self, in_channels=64, channels=3):
        super(FeatureDecouplingModule, self).__init__()
        kernel_size = 1

        w = torch.randn(channels, in_channels, kernel_size, kernel_size)
        self.w0 = torch.nn.Parameter(torch.FloatTensor(
            self.normalize_to_0_1(w)), requires_grad=True)
        w = torch.randn(channels, in_channels, kernel_size, kernel_size)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(
            self.normalize_to_0_1(w)), requires_grad=True)
        w = torch.zeros(channels, in_channels, kernel_size, kernel_size)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(
            w), requires_grad=True)

        self.bias_proportion = torch.nn.Parameter(torch.zeros(
            (channels, 1,  1, 1)), requires_grad=True)

        self.alpha_0 = torch.nn.Parameter(torch.ones(
            (1, channels, 1, 1)), requires_grad=True)
        self.alpha_1 = torch.nn.Parameter(torch.ones(
            (1, channels, 1, 1)), requires_grad=True)
        self.alpha_2 = torch.nn.Parameter(torch.ones(
            (1, channels, 1, 1)), requires_grad=True)

        self.bias_0 = torch.nn.Parameter(torch.zeros(
            (1, channels, 1, 1)), requires_grad=True)
        self.bias_1 = torch.nn.Parameter(torch.zeros(
            (1, channels, 1, 1)), requires_grad=True)
        self.bias_2 = torch.nn.Parameter(torch.zeros(
            (1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        w0 = self.w0
        w1 = self.w1
        o, c, k_w, k_h = w0.shape

        w0_attention = 1+self.w2
        w1_attention = 1-self.w2

        w = w0*w0_attention
        median_w = torch.median(w, dim=1, keepdim=True)
        w0_correct = F.relu(w-median_w.values+self.bias_proportion)

        w0_correct = self.normalize_to_0_1(w0_correct)

        w = w1*w1_attention
        median_w = torch.median(w, dim=1, keepdim=True)
        w1_correct = F.relu(w-median_w.values-self.bias_proportion)
        w1_correct = self.normalize_to_0_1(w1_correct)

        w2_correct = w0_correct+w1_correct

        img = torch.sigmoid(self.alpha_0*F.conv2d(x, w0_correct)+self.bias_0)
        matte = torch.sigmoid(self.alpha_1*F.conv2d(x, w1_correct)+self.bias_1)
        img_free = torch.sigmoid(
            self.alpha_2*F.conv2d(x, w2_correct)+self.bias_2)

        return img, matte, img_free

    def normalize_to_0_1(self, w):
        w = w-w.min()
        w = w/w.max()
        return w


class DMTNSOURCE(nn.Module):
    def __init__(self, in_channels=3, channels=64, norm="batch", stage_num=[6, 4]):
        super(DMTNSOURCE, self).__init__()
        self.stage_num = stage_num

        # Pre-trained VGG19
        self.add_module('vgg19', VGG19())

        # SE
        cat_channels = in_channels+64+128+256+512+512
        self.se = nn.Sequential(SELayer(cat_channels),
                                conv2d_layer(cat_channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm))

        self.down_sample = conv2d_layer(
            channels, 2*channels, kernel_size=4, stride=2, padding=1, dilation=1, norm=norm)

        # coarse
        coarse_list = []
        for i in range(self.stage_num[0]):
            coarse_list.append(SemiConvModule(
                2*channels, norm, mid_dilation=2**(i % 6)))
        self.coarse_list = nn.Sequential(*coarse_list)

        self.up_conv = conv2d_layer(
            2*channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, norm=norm)

        # fine
        fine_list = []
        for i in range(self.stage_num[1]):
            fine_list.append(SemiConvModule(
                channels, norm, mid_dilation=2**(i % 6)))
        self.fine_list = nn.Sequential(*fine_list)

        self.se_coarse = nn.Sequential(SELayer(2*channels),
                                       conv2d_layer(2*channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm))

        # SPP
        self.spp = SPP(channels, norm=norm)

        # Shadow feature decoupling module'
        self.FDM = FeatureDecouplingModule(in_channels=channels, channels=3)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])

        # vgg
        x_vgg = self.vgg19(x)

        # hyper-column features
        x_cat = torch.cat((
            x,
            F.interpolate(x_vgg['relu1_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu2_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu3_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu4_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu5_2'], size, mode="bilinear", align_corners=True)), dim=1)

        # SE
        x = self.se(x_cat)

        # coarse
        x_ = x
        x = self.down_sample(x)
        for i in range(self.stage_num[0]):
            x = self.coarse_list[i](x)

        size = (x_.shape[2], x_.shape[3])
        x = F.interpolate(x, size, mode="bilinear", align_corners=True)
        x = self.up_conv(x)

        # fine
        x = self.se_coarse(torch.cat((x_, x), dim=1))
        for i in range(self.stage_num[1]):
            x = self.fine_list[i](x)

        # spp
        x = self.spp(x)

        # output
        img, matte_out, img_free = self.FDM(x)

        return [img, matte_out, img_free]


class SemiConvModule(nn.Module):
    def __init__(self, channels=64, norm="batch", mid_dilation=2):
        super(SemiConvModule, self).__init__()
        list_factor = solve_factor(channels)
        self.group = list_factor[int(len(list_factor)/2)-1]
        self.split_channels = int(channels/2)

        # Conv
        self.conv_dilation = conv2d_layer(
            self.split_channels, self.split_channels, kernel_size=3,  padding=mid_dilation, dilation=mid_dilation, norm=norm)
        self.conv_3x3 = conv2d_layer(
            self.split_channels, self.split_channels, kernel_size=3,  padding=1, dilation=1, norm=norm)

    def forward(self, x):
        SSRD=False
        if SSRD:
            x_conv = x[:, self.split_channels:, :, :]
            x_identity = x[:, 0:self.split_channels, :, :]
        else:
            x_conv = x[:, 0:self.split_channels, :, :]
            x_identity = x[:, self.split_channels:, :, :]

        x_conv = x_conv+self.conv_dilation(x_conv)+self.conv_3x3(x_conv)

        x = torch.cat((x_identity, x_conv), dim=1)
        x = self.channel_shuffle(x)
        return x

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def identity(self, x):
        return x


class SPP(nn.Module):
    def __init__(self, channels=64, norm="batch"):
        super(SPP, self).__init__()
        self.net2 = avgcov2d_layer(
            4, 4, channels, channels, 1, padding=0, norm=norm)
        self.net8 = avgcov2d_layer(
            8, 8, channels, channels, 1, padding=0, norm=norm)
        self.net16 = avgcov2d_layer(
            16, 16, channels, channels, 1, padding=0, norm=norm)
        self.net32 = avgcov2d_layer(
            32, 32, channels, channels, 1, padding=0, norm=norm)
        self.output = conv2d_layer(channels*5, channels, 3, norm=norm)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = torch.cat((
            F.interpolate(self.net2(x), size, mode="bilinear",
                          align_corners=True),
            F.interpolate(self.net8(x), size, mode="bilinear",
                          align_corners=True),
            F.interpolate(self.net16(x), size,
                          mode="bilinear", align_corners=True),
            F.interpolate(self.net32(x), size,
                          mode="bilinear", align_corners=True),
            x), dim=1)
        x = self.output(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
