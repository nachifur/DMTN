import multiprocessing
from decimal import *

import numpy as np
import skimage.color as skcolor
import torch
import torch.nn as nn


class Metrics(nn.Module):
    def __init__(self):
        super(Metrics, self).__init__()
        nThreads = multiprocessing.cpu_count()
        self.multiprocessingi_utils = MultiprocessingiUtils(int(nThreads/4))

    def rgb2lab_all(self, outputs, GT):
        outputs = self.multiprocessingi_utils.rgb2lab(outputs)
        GT = self.multiprocessingi_utils.rgb2lab(GT)

        return outputs, GT

    def rmse(self, outputs, mask, GT, dataset_mode=0):
        outputs, GT = self.rgb2lab_all(outputs, GT)

        outputs[outputs > 1.0] = 1.0
        outputs[outputs < 0] = 0

        mask[mask > 0] = 1
        mask[mask == 0] = 0
        mask = mask.expand(-1, 3, -1, -1)
        mask_inverse = 1 - mask

        error_map = (outputs - GT).abs()*255

        rmse_all = error_map.sum(dim=(1, 2, 3))
        n_pxl_all = torch.from_numpy(np.array(
            [error_map.shape[2]*error_map.shape[3]])).cuda().expand(rmse_all.shape[0]).type(torch.float32)

        rmse_shadow = (error_map*mask).sum(dim=(1, 2, 3))
        n_pxl_shadow = mask.sum(dim=(1, 2, 3)) / mask.shape[1]

        rmse_non_shadow = (error_map*mask_inverse).sum(dim=(1, 2, 3))
        n_pxl_non_shadow = mask_inverse.sum(
            dim=(1, 2, 3)) / mask_inverse.shape[1]

        if dataset_mode != 0:
            return rmse_shadow, n_pxl_shadow, rmse_non_shadow, n_pxl_non_shadow, rmse_all, n_pxl_all
        else:
            rmse_shadow_eval = (
                rmse_shadow / (n_pxl_shadow+(n_pxl_shadow == 0.0).type(torch.float32))).mean()
            rmse_non_shadow_eval = (
                rmse_non_shadow / (n_pxl_non_shadow+(n_pxl_non_shadow == 0.0).type(torch.float32))).mean()
            rmse_all_eval = (
                rmse_all/(n_pxl_all+(n_pxl_all == 0.0).type(torch.float32))).mean()
            # print('running rmse-shadow: %.4f, rmse-non-shadow: %.4f, rmse-all: %.4f'
            #       % (rmse_shadow_eval, rmse_non_shadow_eval, rmse_all_eval))
            return rmse_shadow_eval, rmse_non_shadow_eval, rmse_all_eval

    def collect_rmse(self, rmse_shadow, n_pxl_shadow, rmse_non_shadow, n_pxl_non_shadow, rmse_all, n_pxl_all):
        # GPU->CPU
        rmse_shadow = rmse_shadow.cpu().numpy()
        n_pxl_shadow = n_pxl_shadow.cpu().numpy()
        rmse_non_shadow = rmse_non_shadow.cpu().numpy()
        n_pxl_non_shadow = n_pxl_non_shadow.cpu().numpy()
        rmse_all = rmse_all.cpu().numpy()
        n_pxl_all = n_pxl_all.cpu().numpy()

        # decimal
        getcontext().prec = 50

        # sum
        rmse_shadow_ = Decimal(0)
        for add in rmse_shadow:
            rmse_shadow_ += Decimal(float(add))
        n_pxl_shadow_ = Decimal(0)
        for add in n_pxl_shadow:
            n_pxl_shadow_ += Decimal(float(add))
        rmse_non_shadow_ = Decimal(0)
        for add in rmse_non_shadow:
            rmse_non_shadow_ += Decimal(float(add))
        n_pxl_non_shadow_ = Decimal(0)
        for add in n_pxl_non_shadow:
            n_pxl_non_shadow_ += Decimal(float(add))
        rmse_all_ = Decimal(0)
        for add in rmse_all:
            rmse_all_ += Decimal(float(add))
        n_pxl_all_ = Decimal(0)
        for add in n_pxl_all:
            n_pxl_all_ += Decimal(float(add))

        # compute
        rmse_shadow_eval = rmse_shadow_ / n_pxl_shadow_
        rmse_non_shadow_eval = rmse_non_shadow_ / n_pxl_non_shadow_
        rmse_all_eval = rmse_all_ / n_pxl_all_

        self.multiprocessingi_utils.close()

        return float(rmse_shadow_eval), float(rmse_non_shadow_eval), float(rmse_all_eval)


def rgb2lab(img):
    img = img.transpose([1, 2, 0])
    return skcolor.rgb2lab(img)


class MultiprocessingiUtils():
    def __init__(self, nThreads=4):
        self.nThreads = nThreads
        self.pool = multiprocessing.Pool(processes=self.nThreads)

    def rgb2lab(self, inputs):
        inputs = inputs.cpu().numpy()
        inputs_ = self.pool.map(rgb2lab, inputs)
        i = 0
        for input_ in inputs_:
            inputs[i, :, :, :] = input_.transpose([2, 0, 1])/255
            i += 1
        output = torch.from_numpy(inputs).cuda()
        return output

    def close(self):
        self.pool.close()
        self.pool.join()

    def creat_pool(self):
        self.pool = multiprocessing.Pool(processes=self.nThreads)
