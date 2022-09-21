import os
import random
import shutil
import sys
import time
from pathlib import Path
from shutil import copyfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

def solve_factor(num):
    # solve factors for a number
    list_factor = []
    i = 1
    if num > 2:
        while i <= num:
            i += 1
            if num % i == 0:
                list_factor.append(i)
            else:
                pass
    else:
        pass

    list_factor = list(set(list_factor))
    list_factor = np.sort(list_factor)
    return list_factor

def copypth(dest_path, src_path):
    if (src_path).is_file():
        copyfile(src_path, dest_path)
        print(str(src_path)+" copy to "+str(dest_path))


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    # https://github.com/dojure/FPIE/blob/master/utils.py
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis=1)
    return out_filter


def blur(x, kernel_var):
    return F.conv2d(x, kernel_var, padding=1)


def sobel_kernel(kernlen=3, channels=1):
    out_filter = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter_x = np.repeat(out_filter, channels, axis=1)

    out_filter = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter_y = np.repeat(out_filter, channels, axis=1)
    return [out_filter_x, out_filter_y]


def sobel(x, kernel_var):
    sobel_kernel_x = kernel_var[0]
    sobel_kernel_y = kernel_var[1]
    sobel_x = F.conv2d(x, sobel_kernel_x, padding=1)
    sobel_y = F.conv2d(x, sobel_kernel_y, padding=1)

    return sobel_x.abs()+sobel_y.abs()


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def create_config(config_path, cover=False):
    if cover:
        copyfile('./config.yml', config_path)
    else:
        if not os.path.exists(config_path):
            copyfile('./config.yml', config_path)


def init_config(checkpoints_path, MODE=0, EVAL_INTERVAL_EPOCH=1, EPOCH=[30, 0]):
    if EVAL_INTERVAL_EPOCH < 1:
        APPEND = 0
    else:
        APPEND = 1
    if len(EPOCH) > 2:
        lr_restart = True
    else:
        lr_restart = False
    # Re-training after training is interrupted abnormally
    skip_train = restart_train(
        checkpoints_path, MODE, APPEND, EPOCH, lr_restart)
    if skip_train:
        return skip_train

    # edit config
    config_path = os.path.join(checkpoints_path, 'config.yml')
    fr = open(config_path, 'r')
    config = yaml.load(fr, Loader=yaml.FullLoader)
    fr.close()

    EPOCHLIST = EPOCH
    ALL_EPOCH = EPOCH[-2]
    EPOCH = EPOCH[EPOCH[-1]]

    if MODE == 0 or MODE == 1:
        flist = config['TRAIN_FLIST_PRE']
    else:
        flist = config['TRAIN_FLIST']
    TRAIN_DATA_NUM = len(np.genfromtxt(
        config["DATA_ROOT"]+flist, dtype=np.str, encoding='utf-8'))
    print('train data number is:{}'.format(TRAIN_DATA_NUM))

    if torch.cuda.is_available():
        BATCH_SIZE = config['BATCH_SIZE']
        MAX_ITERS = EPOCH * (TRAIN_DATA_NUM // BATCH_SIZE) # drop_last=True
        if config["DEBUG"]:
            INTERVAL = 10
            config['MAX_ITERS'] = 80
            config['EVAL_INTERVAL'] = INTERVAL
            config['DEBUG'] = 1
            config['SAMPLE_INTERVAL'] = INTERVAL
            config['SAVE_INTERVAL'] = INTERVAL
        else:
            INTERVAL = ((EVAL_INTERVAL_EPOCH * (TRAIN_DATA_NUM // BATCH_SIZE))//10)*10
            config['MAX_ITERS'] = MAX_ITERS
            config['EVAL_INTERVAL'] = INTERVAL
            config['DEBUG'] = 0
            config['SAMPLE_INTERVAL'] = 500
            config['SAVE_INTERVAL'] = INTERVAL

        if EVAL_INTERVAL_EPOCH < 1:
            config['FORCE_EXIT'] = 0
        else:
            config['FORCE_EXIT'] = 1

        config["MODE"] = MODE
        config['ALL_EPOCH'] = ALL_EPOCH
        config['EPOCH'] = EPOCH
        config['EPOCHLIST'] = EPOCHLIST
        config['APPEND'] = APPEND
        config['EVAL_INTERVAL_EPOCH'] = EVAL_INTERVAL_EPOCH
        save_config(config, config_path)
    else:
        print("cuda is unavailable")

    return skip_train


def restart_train(checkpoints_path, MODE, APPEND, EPOCH, lr_restart):
    config_path = os.path.join(checkpoints_path, 'config.yml')
    fr = open('./config.yml', 'r')
    config = yaml.load(fr, Loader=yaml.FullLoader)
    fr.close()

    if MODE == 0:
        src_path = Path('./pre_train_model') / \
            config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'_pre_da.pth')
    elif MODE == 1:
        src_path = Path('./pre_train_model') / \
            config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'_pre_no_da.pth')
    elif MODE == 2:
        src_path = Path('./pre_train_model') / \
            config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'_final.pth')

    if lr_restart:
        src_path = Path(checkpoints_path)/("model_save_mode_" +
                                           str(MODE))/(str(EPOCH[EPOCH[-1]]-1)+".0.pth")

    log_eval_val_ap_path = Path(checkpoints_path) / \
        ('log_eval_val_ap_'+str(MODE)+'.txt')



    if Path(src_path).is_file():
        skip_train = True
        cover = False  # retrain
    else:
        skip_train = False
        if APPEND == 1 and log_eval_val_ap_path.is_file():
            cover = False
        else:
            cover = True  # new train stage

    # append
    if (not Path(src_path).is_file()) and APPEND == 1 and log_eval_val_ap_path.is_file():
        eval_val_ap = np.genfromtxt(
            log_eval_val_ap_path, dtype=np.str, encoding='utf-8').astype(np.float)
        src_path = Path(checkpoints_path)/("model_save_mode_" +
                                           str(MODE))/(str(eval_val_ap[0, 1]) + '.pth')

        if eval_val_ap[0, 1]==EPOCH[EPOCH[-1]-1]-1:
            cover = True

    # copy .pth
    dest_path = Path('checkpoints/') / \
        config["SUBJECT_WORD"]/(config["MODEL_NAME"]+'.pth')
    copypth(dest_path, src_path)

    # create config
    create_config(config_path, cover=cover)
    print("cover config file-"+str(cover))

    if skip_train:
        print("skip train stage of mode-"+str(MODE))
    return skip_train


def save_config(config, config_path):
    with open(config_path, 'w') as f_obj:
        yaml.dump(config, f_obj)


def stitch_images(*outputs, img_per_row=2):
    inputs_all = [*outputs]
    inputs = inputs_all[0]
    gap = 5
    images = [inputs_all[0], *inputs_all[1], *inputs_all[2:]]

    columns = len(images)

    height, width = inputs[0][:, :, 0].shape
    img = Image.new('RGB', (width * img_per_row * columns + gap *
                            (img_per_row - 1), height * int(len(inputs) / img_per_row)))

    for ix in range(len(inputs)):
        xoffset = int(ix % img_per_row) * width * \
            columns + int(ix % img_per_row) * gap
        yoffset = int(ix / img_per_row) * height

        for cat in range(len(images)):
            im = np.array((images[cat][ix]).cpu()).astype(np.uint8).squeeze()
            im = Image.fromarray(im)
            img.paste(im, (xoffset + cat * width, yoffset))

    return img


def imshow(img, title=''):
    fig = plt.gcf()
    fig.canvas.set_window_title(title)
    plt.axis('off')
    if len(img.size) == 3:
        plt.imshow(img, interpolation='none')
        plt.show()
    else:
        plt.imshow(img, cmap='Greys_r')
        plt.show()


def imsave(img, path):
    im = Image.fromarray(img.cpu().numpy().astype(np.uint8).squeeze())
    im.save(path)


def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    # # https://github.com/knazeri/edge-connect
    mask = np.zeros((height, width))
    mask_x = x if x is not None else random.randint(0, width - mask_width)
    mask_y = y if y is not None else random.randint(0, height - mask_height)
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1
    return mask


class Progbar(object):
    # https://github.com/knazeri/edge-connect
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules or
                                 'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                   (eta % 3600) // 60,
                                                   eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
