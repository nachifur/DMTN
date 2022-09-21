import argparse
import os
import random
from shutil import copyfile

import cv2
import numpy as np
import torch

from src.config import Config
from src.model_top import ModelTop
from src.utils import create_dir


def main(mode, config_path):
    r"""
    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode, config_path)
    config.CONFIG_PATH = config_path

    # init device
    if torch.cuda.is_available():
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)

    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = ModelTop(config)
    model.load()

    # model pre training (data augmentation)
    if config.MODE == 0:
        config.print()
        model.train()

    # model pre training (no data augmentation)
    if config.MODE == 1:
        config.print()
        model.train()

    # model training
    if config.MODE == 2:
        config.print()
        model.train()

    # model test
    elif config.MODE == 3:
        model.test()

    # model eval on val set
    elif config.MODE ==4:
        model.eval()

    # model eval on test set
    elif config.MODE == 5:
        model.eval()


def load_config(mode, config_path):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test 3:eval reads from config file if not specified
    """

    # load config file
    config = Config(config_path)
    config.MODE = mode

    return config


if __name__ == "__main__":
    main()
