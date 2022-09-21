import os

import torch
import torch.nn as nn
import torch.optim as optim
from src.network.network_DMTN import DMTN

class BaseModel(nn.Module):
    def __init__(self, name, config):
        super(BaseModel, self).__init__()

        self.name = name
        self.config = config
        self.iteration = 0
        self.config = config

        self.weights_path = os.path.join(config.PATH, name + '.pth')

    def load(self):
        if os.path.exists(self.weights_path):
            print('Loading %s model...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.weights_path)
                print(self.weights_path)
            else:
                data = torch.load(self.weights_path,
                                  map_location=lambda storage, loc: storage)
            self.network_instance.load_state_dict(data['model'])
            self.iteration = data['iteration']
        else:
            vgg19_weights_path = self.config.DATA_ROOT+"/vgg19-dcbb9e9d.pth"
            self.network_instance.network.vgg19.load_pretrained(
                vgg19_weights_path, self.config.DEVICE)
            for discriminator in self.network_instance.discriminator:
                discriminator.perceptual_loss.vgg19.load_pretrained(
                    vgg19_weights_path, self.config.DEVICE)

    def save(self):
        print('\nsaving %s...\n' % self.weights_path)
        torch.save({
            'iteration': self.iteration,
            'model': self.network_instance.state_dict()
        }, self.weights_path)

        if self.config.BACKUP:
            INTERVAL_ = 4
            if self.config.SAVE_INTERVAL and self.iteration % (self.config.SAVE_INTERVAL*INTERVAL_) == 0:
                print('\nsaving %s...\n' % self.name+'_backup')
                torch.save({
                    'iteration': self.iteration,
                    'model': self.network_instance.state_dict()
                }, os.path.join(self.config.PATH, 'backups/' + self.name + '_' + str(self.iteration // (self.config.SAVE_INTERVAL*INTERVAL_)) + '.pth'))


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config.MODEL_NAME, config)
        self.INNER_OPTIMIZER = config.INNER_OPTIMIZER
        # networks choose
        if config.NETWORK == "DMTN":
            network_instance = DMTN(config, in_channels=3)
        else:
            network_instance = None
        self.add_module('network_instance', network_instance)

    def process(self, images, mask, GT, eval_mode=False):
        if not eval_mode:
            self.iteration += 1
        outputs, loss, logs = self.network_instance.process(
            images, mask, GT)

        return outputs, loss, logs

    def forward(self, images):
        inputs = images
        edges = self.network_instance(inputs)
        return edges

    def backward(self, loss=None):
        self.network_instance.backward(loss)
