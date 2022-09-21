import multiprocessing
import os
from pathlib import Path

import numpy as np
import yaml


from src.main import main
from src.utils import create_dir, init_config, copypth
import torch

if __name__ == '__main__':
    # inital
    multiprocessing.set_start_method('spawn')
    config = yaml.load(open("config.yml", 'r'), Loader=yaml.FullLoader)
    dest_path = Path('checkpoints/') / \
        config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'.pth')
    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        str(e) for e in config["GPU"])
    torch.autograd.set_detect_anomaly(True)
    checkpoints_path = Path('./checkpoints') / \
        config["SUBJECT_WORD"]  # model checkpoints path
    create_dir(checkpoints_path)
    create_dir('./pre_train_model')
    config_path = os.path.join(checkpoints_path, 'config.yml')

    # pre_train (no data augmentation)
    MODE = 0
    print('\nmode-'+str(MODE)+': start pre_training(data augmentation)...\n')
    for i in range(1):
        skip_train = init_config(checkpoints_path, MODE=MODE,
                                EVAL_INTERVAL_EPOCH=1, EPOCH=[90,i])
        if not skip_train:
            main(MODE, config_path)
    src_path = Path('./pre_train_model') / \
        config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'_pre_da.pth')
    copypth(dest_path, src_path)

    # train
    MODE = 2
    print('\nmode-'+str(MODE)+': start training...\n')
    for i in range(1):
        skip_train = init_config(checkpoints_path, MODE=MODE,
                                EVAL_INTERVAL_EPOCH=0.1, EPOCH=[60,i])
        if not skip_train:
            main(MODE, config_path)
    src_path = Path('./pre_train_model') / \
        config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'_final.pth')
    copypth(dest_path, src_path)

    # test
    MODE = 3
    print('\nmode-'+str(MODE)+': start testing...\n')
    main(MODE, config_path)

    # eval on val set
    # MODE = 4
    # print('\nmode-'+str(MODE)+': start eval...\n')
    # main(MODE,config_path)

    # eval on test set
    MODE = 5
    print('\nmode-'+str(MODE)+': start eval...\n')
    main(MODE, config_path)
