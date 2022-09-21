"""
This part of the code is built based on the project:
https://github.com/knazeri/edge-connect
"""
import torch
import torch.nn as nn
import torchvision.models as models
from .utils import blur, gauss_kernel, sobel, sobel_kernel
from torch.autograd import Variable
import numpy as np
import os


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss(reduction="mean")

        elif type == 'lsgan':
            self.criterion = nn.MSELoss(reduction="mean")

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            loss = self.criterion(outputs, labels)
            return loss


class StyleLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.add_module('vgg19', VGG19())
        self.criterion = torch.nn.L1Loss(reduction="mean")

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)

        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg19(x), self.vgg19(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(
            x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(
            x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(
            x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(
            x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class PerceptualLoss(nn.Module):
    r"""
    Perceptual loss, VGG-based
    https://arxiv.org/abs/1603.08155
    https://github.com/dxyang/StyleTransfer/blob/master/utils.py
    """

    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.add_module('vgg19', VGG19())
        self.criterion = torch.nn.L1Loss(reduction="mean")
        self.weights = weights

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg19(x), self.vgg19(y)

        p1 = self.weights[0] * \
            self.criterion(x_vgg['relu1_2'], y_vgg['relu1_2'])
        p2 = self.weights[1] * \
            self.criterion(x_vgg['relu2_2'], y_vgg['relu2_2'])
        p3 = self.weights[2] * \
            self.criterion(x_vgg['relu3_2'], y_vgg['relu3_2'])
        p4 = self.weights[3] * \
            self.criterion(x_vgg['relu4_2'], y_vgg['relu4_2'])
        p5 = self.weights[4] * \
            self.criterion(x_vgg['relu5_2'], y_vgg['relu5_2'])

        return p1+p2+p3+p4+p5


class VGG19(torch.nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        # https://pytorch.org/hub/pytorch_vision_vgg/
        mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32)
        mean = mean.reshape((1, 3, 1, 1))
        self.mean = Variable(torch.from_numpy(mean)).cuda()
        std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
        std = std.reshape((1, 3, 1, 1))
        self.std = Variable(torch.from_numpy(std)).cuda()
        self.initial_model()

    def forward(self, x):
        relu1_1 = self.relu1_1((x-self.mean)/self.std)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

    def load_pretrained(self, vgg19_weights_path, gpu):
        if os.path.exists(vgg19_weights_path):
            if torch.cuda.is_available():
                data = torch.load(vgg19_weights_path)
                print("load vgg_pretrained_model:"+vgg19_weights_path)
            else:
                data = torch.load(vgg19_weights_path,
                                  map_location=lambda storage, loc: storage)
            self.initial_model(data)
            self.to(gpu)
        else:
            print("you need download vgg_pretrained_model in the directory of  "+str(self.config.DATA_ROOT) +
                  "\n'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'")
            raise Exception("Don't load vgg_pretrained_model")

    def initial_model(self,data=None):
            vgg19 = models.vgg19()
            if data is not None:
                vgg19.load_state_dict(data)
            features = vgg19.features
            self.relu1_1 = torch.nn.Sequential()
            self.relu1_2 = torch.nn.Sequential()

            self.relu2_1 = torch.nn.Sequential()
            self.relu2_2 = torch.nn.Sequential()

            self.relu3_1 = torch.nn.Sequential()
            self.relu3_2 = torch.nn.Sequential()
            self.relu3_3 = torch.nn.Sequential()
            self.relu3_4 = torch.nn.Sequential()

            self.relu4_1 = torch.nn.Sequential()
            self.relu4_2 = torch.nn.Sequential()
            self.relu4_3 = torch.nn.Sequential()
            self.relu4_4 = torch.nn.Sequential()

            self.relu5_1 = torch.nn.Sequential()
            self.relu5_2 = torch.nn.Sequential()
            self.relu5_3 = torch.nn.Sequential()
            self.relu5_4 = torch.nn.Sequential()

            for x in range(2):
                self.relu1_1.add_module(str(x), features[x])

            for x in range(2, 4):
                self.relu1_2.add_module(str(x), features[x])

            for x in range(4, 7):
                self.relu2_1.add_module(str(x), features[x])

            for x in range(7, 9):
                self.relu2_2.add_module(str(x), features[x])

            for x in range(9, 12):
                self.relu3_1.add_module(str(x), features[x])

            for x in range(12, 14):
                self.relu3_2.add_module(str(x), features[x])

            for x in range(14, 16):
                self.relu3_3.add_module(str(x), features[x])

            for x in range(16, 18):
                self.relu3_4.add_module(str(x), features[x])

            for x in range(18, 21):
                self.relu4_1.add_module(str(x), features[x])

            for x in range(21, 23):
                self.relu4_2.add_module(str(x), features[x])

            for x in range(23, 25):
                self.relu4_3.add_module(str(x), features[x])

            for x in range(25, 27):
                self.relu4_4.add_module(str(x), features[x])

            for x in range(27, 30):
                self.relu5_1.add_module(str(x), features[x])

            for x in range(30, 32):
                self.relu5_2.add_module(str(x), features[x])

            for x in range(32, 34):
                self.relu5_3.add_module(str(x), features[x])

            for x in range(34, 36):
                self.relu5_4.add_module(str(x), features[x])

            # don't need the gradients, just want the features
            # for param in self.parameters():
            #     param.requires_grad = False