import os
from pathlib import Path
from shutil import copyfile

import numpy as np
import torch
import torchvision.transforms.functional as F
import yaml
from torch.utils.data import DataLoader

from .dataset import Dataset
from .metrics import Metrics
from .models import Model
from .utils import (Progbar, create_dir, imsave, imshow, save_config,
                    stitch_images)


class ModelTop():
    def __init__(self, config):
        # config
        self.config = config
        if config.DEBUG == 1:
            self.debug = True
        else:
            self.debug = False
        self.model_name = config.MODEL_NAME
        self.RESULTS_SAMPLE = self.config.RESULTS_SAMPLE
        # model
        self.model = Model(config).to(config.DEVICE)
        # eval
        self.metrics = Metrics().to(config.DEVICE)
        # dataset
        if config.MODE == 3:  # test
            self.test_dataset = Dataset(
                config, config.DATA_ROOT+config.TEST_FLIST, config.DATA_ROOT+config.TEST_MASK_FLIST, config.DATA_ROOT+config.TEST_GT_FLIST, augment=False)
        elif config.MODE == 4:  # eval
            self.val_dataset = Dataset(
                config, config.DATA_ROOT+config.VAL_FLIST, config.DATA_ROOT+config.VAL_MASK_FLIST, config.DATA_ROOT+config.VAL_GT_FLIST, augment=False)
        elif config.MODE == 5:  # eval
            self.val_dataset = Dataset(
                config, config.DATA_ROOT+config.TEST_FLIST, config.DATA_ROOT+config.TEST_MASK_FLIST, config.DATA_ROOT+config.TEST_GT_FLIST, augment=False)
        else:
            if config.MODE == 0:
                self.train_dataset = Dataset(
                    config, config.DATA_ROOT+config.TRAIN_FLIST_PRE, config.DATA_ROOT+config.TRAIN_MASK_FLIST_PRE, config.DATA_ROOT+config.TRAIN_GT_FLIST_PRE, augment=True)
                self.val_dataset = Dataset(
                    config, config.DATA_ROOT+config.VAL_FLIST_PRE, config.DATA_ROOT+config.VAL_MASK_FLIST_PRE, config.DATA_ROOT+config.VAL_GT_FLIST_PRE, augment=True)
            elif config.MODE == 1:
                self.train_dataset = Dataset(
                    config, config.DATA_ROOT+config.TRAIN_FLIST_PRE, config.DATA_ROOT+config.TRAIN_MASK_FLIST_PRE, config.DATA_ROOT+config.TRAIN_GT_FLIST_PRE, augment=False)
                self.val_dataset = Dataset(
                    config, config.DATA_ROOT+config.VAL_FLIST_PRE, config.DATA_ROOT+config.VAL_MASK_FLIST_PRE, config.DATA_ROOT+config.VAL_GT_FLIST_PRE, augment=False)
            elif config.MODE == 2:
                self.train_dataset = Dataset(
                    config, config.DATA_ROOT+config.TRAIN_FLIST, config.DATA_ROOT+config.TRAIN_MASK_FLIST, config.DATA_ROOT+config.TRAIN_GT_FLIST, augment=False)
                self.val_dataset = Dataset(
                    config, config.DATA_ROOT+config.VAL_FLIST, config.DATA_ROOT+config.VAL_MASK_FLIST, config.DATA_ROOT+config.VAL_GT_FLIST, augment=False)
            self.sample_iterator = self.val_dataset.create_iterator(
                config.SAMPLE_SIZE)

        # path
        self.samples_path = os.path.join(config.PATH, 'samples')
        self.results_path = os.path.join(config.PATH, 'results')
        self.backups_path = os.path.join(config.PATH, 'backups')
        self.results_samples_path = os.path.join(self.results_path, 'samples')
        if self.config.BACKUP:
            create_dir(self.backups_path)
        if config.RESULTS is not None:
            self.results_path = os.path.join(config.RESULTS)
        create_dir("./pre_train_model/"+self.config.SUBJECT_WORD)
        # load file
        self.log_file = os.path.join(
            config.PATH, 'log_' + self.model_name + '.dat')

        # avoid overfitting
        if config.MODE < 3:
            data_save_path = os.path.join(
                self.config.PATH, 'log_eval_val_ap_id.txt')
            if os.path.exists(data_save_path):
                self.eval_val_ap_id = np.genfromtxt(
                    data_save_path, dtype=np.str, encoding='utf-8').astype(np.float)
                if config.APPEND == 0:
                    self.eval_val_ap_id[config.MODE] = 0
            else:
                self.eval_val_ap_id = [0.0, 0.0, 0.0]

            data_save_path = os.path.join(
                self.config.PATH, 'final_model_epoch.txt')
            if os.path.exists(data_save_path):
                self.epoch = np.genfromtxt(
                    data_save_path, dtype=np.str, encoding='utf-8').astype(np.float).astype(np.int)
                if config.APPEND == 0:
                    self.epoch[config.MODE, 0] = 0
            else:
                self.epoch = np.zeros((3, 2)).astype(int)

            data_save_path = os.path.join(
                self.config.PATH, 'log_eval_val_ap_'+str(self.config.MODE)+'.txt')
            if os.path.exists(data_save_path):
                self.eval_val_ap = np.genfromtxt(
                    data_save_path, dtype=np.str, encoding='utf-8').astype(np.float)
                if config.APPEND == 0:
                    if config.FORCE_EXIT == 0:
                        if config.MODE == 0 or config.MODE == 1:
                            self.eval_val_ap = np.ones(
                                (config.PRE_TRAIN_EVAL_LEN, 2))*1e6
                        elif config.MODE == 2:
                            self.eval_val_ap = np.ones(
                                (config.TRAIN_EVAL_LEN, 2))*1e6
                    else:
                        self.eval_val_ap = np.ones((config.ALL_EPOCH, 2))*1e6
            else:
                if config.FORCE_EXIT == 0:
                    if config.MODE == 0 or config.MODE == 1:
                        self.eval_val_ap = np.ones(
                            (config.PRE_TRAIN_EVAL_LEN, 2))*1e6
                    elif config.MODE == 2:
                        self.eval_val_ap = np.ones(
                            (config.TRAIN_EVAL_LEN, 2))*1e6
                else:
                    self.eval_val_ap = np.ones((config.ALL_EPOCH, 2))*1e6

        # lr scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.model.network_instance.optimizer, 'min', factor=0.5, patience=0, min_lr=1e-5)
        if config.ADV:
            self.scheduler_dis = [torch.optim.lr_scheduler.ReduceLROnPlateau(
                discriminator.optimizer, 'min', factor=0.5, patience=0, min_lr=5e-6) for discriminator in self.model.network_instance.discriminator]

    def load(self):
        self.model.load()

    def save(self):
        self.model.save()

    def train(self):
        # initial
        self.restart_train_check_lr_scheduler()
        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.BATCH_SIZE,
            num_workers=4,
            drop_last=True,
            shuffle=True
        )
        keep_training = True
        mode = self.config.MODE
        max_iteration = int(float((self.config.MAX_ITERS)))
        max_epoch = self.config.EPOCHLIST[self.config.EPOCHLIST[-1]]+1
        total = len(self.train_dataset)
        self.TRAIN_DATA_NUM = total
        if total == 0:
            print(
                'No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return
        # train
        while(keep_training):
            # epoch
            epoch = self.epoch[self.config.MODE, 0]
            epoch += 1
            self.epoch[self.config.MODE, 0] = epoch
            print('\n\nTraining epoch: %d' % epoch)
            # progbar
            progbar = Progbar(
                total, width=20, stateful_metrics=['epoch', 'iter'])

            for items in train_loader:
                # initial
                self.model.train()

                # get data
                images, mask, GT = self.cuda(
                    *items)
                # imshow(F.to_pil_image((outputs)[0,:,:,:].cpu()))
                # train
                outputs, loss, logs = self.model.process(
                    images, mask, GT)
                # backward
                self.model.backward(loss)
                iteration = self.model.iteration

                # log-epoch, iteration
                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs
                # progbar
                progbar.add(len(images), values=logs if self.config.VERBOSE else [
                            x for x in logs if not x[0].startswith('l_')])

                # log model at checkpoints
                if self.config.LOG_INTERVAL and iteration % self.config.LOG_INTERVAL == 0:
                    self.log(logs)
                # sample model at checkpoints
                if self.config.SAMPLE_INTERVAL and iteration % self.config.SAMPLE_INTERVAL == 0:
                    self.sample()
                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0:
                    self.save()
                # force_exit
                if epoch >= max_epoch: # if iteration >= max_iteration:
                    force_exit = True
                else:
                    force_exit = False   
                # end condition
                if force_exit:
                    keep_training = False
                    self.force_exit()
                    print('\n ***force_exit: max iteration***')
                    break
                # evaluate model at checkpoints
                if (self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0):
                    print('\nstart eval...\n')
                    pre_exit, ap = self.eval()
                    self.scheduler.step(ap)
                    if self.config.ADV:
                        for scheduler_dis in self.scheduler_dis:
                            scheduler_dis.step(ap)

                    with open(self.config.CONFIG_PATH, 'r') as f_obj:
                        config = yaml.load(f_obj, Loader=yaml.FullLoader)
                    config['LR'] = self.scheduler.optimizer.param_groups[0]['lr']
                    config['LR_D'] = self.scheduler_dis[0].optimizer.param_groups[0]['lr']
                    save_config(config, self.config.CONFIG_PATH)
                else:
                    pre_exit = False
                # debug
                if self.debug:
                    if iteration >= 40:
                        if self.config.MODE == 0:
                            force_exit = True
                            copyfile(Path('checkpoints')/self.config.SUBJECT_WORD/(self.config.MODEL_NAME+'.pth'),
                                     Path('./pre_train_model')/self.config.SUBJECT_WORD/(self.config.MODEL_NAME+'_pre_da.pth'))
                        if self.config.MODE == 1:
                            force_exit = True
                            copyfile(Path('checkpoints')/self.config.SUBJECT_WORD/(self.config.MODEL_NAME+'.pth'),
                                     Path('./pre_train_model')/self.config.SUBJECT_WORD/(self.config.MODEL_NAME+'_pre_no_da.pth'))
                # end condition
                if pre_exit:
                    keep_training = False
                    break

        print('\nEnd training....')

    def eval(self):
        # torch.cuda.empty_cache()
        if self.config.MODE == 4 or self.config.MODE == 5:
            BATCH_SIZE = self.config.BATCH_SIZE  # *8
            num_workers = 4  # 8
        else:
            BATCH_SIZE = self.config.BATCH_SIZE
            num_workers = 4
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            drop_last=False,
            shuffle=False
        )
        total = len(self.val_dataset)
        self.metrics.multiprocessingi_utils.creat_pool()

        self.model.eval()
        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0
        log_eval_PR = [[0], [0]]
        n_thresh = 99
        # zero all counts
        rmse_shadow = torch.Tensor(0).cuda()
        n_pxl_shadow = torch.Tensor(0).cuda()
        rmse_non_shadow = torch.Tensor(0).cuda()
        n_pxl_non_shadow = torch.Tensor(0).cuda()
        rmse_all = torch.Tensor(0).cuda()
        n_pxl_all = torch.Tensor(0).cuda()
        # eval each image
        with torch.no_grad():
            for items in val_loader:
                iteration += 1
                images, mask, GT = self.cuda(
                    *items)
                # eval
                outputs = self.model(images)

                rmse_shadow_, n_pxl_shadow_, rmse_non_shadow_, n_pxl_non_shadow_, rmse_all_, n_pxl_all_ = self.metrics.rmse(
                    outputs[-1], mask, GT, dataset_mode=1)

                rmse_shadow = torch.cat((rmse_shadow, rmse_shadow_), dim=0)
                n_pxl_shadow = torch.cat((n_pxl_shadow, n_pxl_shadow_), dim=0)
                rmse_non_shadow = torch.cat(
                    (rmse_non_shadow, rmse_non_shadow_), dim=0)
                n_pxl_non_shadow = torch.cat(
                    (n_pxl_non_shadow, n_pxl_non_shadow_), dim=0)
                rmse_all = torch.cat((rmse_all, rmse_all_), dim=0)
                n_pxl_all = torch.cat((n_pxl_all, n_pxl_all_), dim=0)

                if self.debug:
                    if iteration == 10:
                        break

        rmse_shadow_eval, rmse_non_shadow_eval, rmse_all_eval = self.metrics.collect_rmse(
            rmse_shadow, n_pxl_shadow, rmse_non_shadow, n_pxl_non_shadow, rmse_all, n_pxl_all)

        # print
        print('running rmse-shadow: %.4f, rmse-non-shadow: %.4f, rmse-all: %.4f'
              % (rmse_shadow_eval, rmse_non_shadow_eval, rmse_all_eval))
        data_save_path = os.path.join(
            self.config.PATH, 'show_result.txt')
        np.savetxt(data_save_path, [
                   rmse_shadow_eval, rmse_non_shadow_eval, rmse_all_eval], fmt='%s')

        # avoid overfitting (pre_train and train)
        if self.config.MODE == 0 or self.config.MODE == 1 or self.config.MODE == 2:
            if self.config.FORCE_EXIT == 0:
                if rmse_all_eval < np.mean(self.eval_val_ap[:, 0]):
                    exit_ = False
                else:
                    exit_ = True
            else:
                exit_ = False

            self.eval_val_ap = np.delete(self.eval_val_ap, -1, axis=0)
            self.eval_val_ap = np.append(
                [[rmse_all_eval, self.eval_val_ap_id[self.config.MODE]]], self.eval_val_ap, axis=0)

            model_save_path = "model_save_mode_"+str(self.config.MODE)
            model_save_path = Path(self.config.PATH)/model_save_path
            create_dir(model_save_path)
            self.model.weights_path = os.path.join(
                model_save_path, str(self.eval_val_ap_id[self.config.MODE]) + '.pth')
            self.save()
            self.model.weights_path = os.path.join(
                self.config.PATH, self.model_name + '.pth')

            data_save_path = os.path.join(
                self.config.PATH, 'final_model_epoch.txt')
            np.savetxt(data_save_path, self.epoch, fmt='%s')
            if self.eval_val_ap_id[self.config.MODE] == (len(self.eval_val_ap)-1):
                self.eval_val_ap_id[self.config.MODE] = 0.0
            else:
                self.eval_val_ap_id[self.config.MODE] += 1.0

            data_save_path = os.path.join(
                self.config.PATH, 'log_eval_val_ap_'+str(self.config.MODE)+'.txt')
            np.savetxt(data_save_path, self.eval_val_ap, fmt='%s')
            data_save_path = os.path.join(
                self.config.PATH, 'log_eval_val_ap_id.txt')
            np.savetxt(data_save_path, self.eval_val_ap_id, fmt='%s')

            if exit_:
                idmin = np.array(self.eval_val_ap[:, 0]).argmin()

                data_save_path = os.path.join(
                    self.config.PATH, 'final_model_epoch.txt')
                if self.config.EVAL_INTERVAL_EPOCH < 1:
                    self.epoch[self.config.MODE,
                               1] = self.eval_val_ap[idmin, 1]
                else:
                    self.epoch[self.config.MODE,
                               1] = self.epoch[self.config.MODE, 0]-idmin-1
                print('final model id:'+str(self.epoch[self.config.MODE, 1]))
                np.savetxt(data_save_path, self.epoch, fmt='%s')

                pre_train_save_path = "./pre_train_model/"+self.config.SUBJECT_WORD
                if os.path.exists(os.path.join(model_save_path, str(self.eval_val_ap[idmin, 1]) + '.pth')):
                    if self.config.MODE == 0:
                        PATH_WEIDHT = os.path.join(
                            pre_train_save_path, self.model_name + '_pre_da.pth')
                    elif self.config.MODE == 1:
                        PATH_WEIDHT = os.path.join(
                            pre_train_save_path, self.model_name + '_pre_no_da.pth')
                    elif self.config.MODE == 2:
                        PATH_WEIDHT = os.path.join(
                            pre_train_save_path, self.model_name + '_final.pth')
                    copyfile(os.path.join(model_save_path, str(
                        self.eval_val_ap[idmin, 1]) + '.pth'), PATH_WEIDHT)
                    print(os.path.join(model_save_path, str(
                        self.eval_val_ap[idmin, 1]) + '.pth')+" copy to "+PATH_WEIDHT)
                    pre_model_save = torch.load(PATH_WEIDHT)
                    torch.save(
                        {'iteration': 0, 'model': pre_model_save['model']}, PATH_WEIDHT)

                    copyfile(PATH_WEIDHT, os.path.join(
                        self.config.PATH, self.model_name + '.pth'))
            return exit_, rmse_all_eval

    def force_exit(self):
        model_save_path = "model_save_mode_"+str(self.config.MODE)
        model_save_path = Path(self.config.PATH)/model_save_path

        idmin = np.array(self.eval_val_ap[:, 0]).argmin()

        data_save_path = os.path.join(
            self.config.PATH, 'final_model_epoch.txt')

        self.epoch[self.config.MODE,0] = self.epoch[self.config.MODE,0]-1 
        if self.config.EVAL_INTERVAL_EPOCH < 1:
            self.epoch[self.config.MODE,
                        1] = self.eval_val_ap[idmin, 1]
        else:
            self.epoch[self.config.MODE,
                        1] = self.epoch[self.config.MODE, 0]-idmin-1
        print('\n\nfinal model id:'+str(self.epoch[self.config.MODE, 1]))
        np.savetxt(data_save_path, self.epoch, fmt='%s')

        pre_train_save_path = "./pre_train_model/"+self.config.SUBJECT_WORD
        if os.path.exists(os.path.join(model_save_path, str(self.eval_val_ap[idmin, 1]) + '.pth')):
            if self.config.MODE == 0:
                PATH_WEIDHT = os.path.join(
                    pre_train_save_path, self.model_name + '_pre_da.pth')
            elif self.config.MODE == 1:
                PATH_WEIDHT = os.path.join(
                    pre_train_save_path, self.model_name + '_pre_no_da.pth')
            elif self.config.MODE == 2:
                PATH_WEIDHT = os.path.join(
                    pre_train_save_path, self.model_name + '_final.pth')
            copyfile(os.path.join(model_save_path, str(
                self.eval_val_ap[idmin, 1]) + '.pth'), PATH_WEIDHT)
            print(os.path.join(model_save_path, str(
                self.eval_val_ap[idmin, 1]) + '.pth')+" copy to "+PATH_WEIDHT)
            pre_model_save = torch.load(PATH_WEIDHT)
            torch.save(
                {'iteration': 0, 'model': pre_model_save['model']}, PATH_WEIDHT)

            copyfile(PATH_WEIDHT, os.path.join(
                self.config.PATH, self.model_name + '.pth'))

    def test(self):
        # initial
        self.model.eval()
        if self.RESULTS_SAMPLE:
            save_path = os.path.join(
                self.results_samples_path, self.model_name)
            create_dir(save_path)
        else:
            save_path = os.path.join(self.results_path, self.model_name)
            create_dir(save_path)
        if self.debug:
            debug_path = os.path.join(save_path, "debug")
            create_dir(debug_path)
            save_path = debug_path
        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )
        # test
        index = 0
        with torch.no_grad():
            for items in test_loader:
                name = self.test_dataset.load_name(index)
                index += 1
                images, mask, GT = self.cuda(
                    *items)
                if self.RESULTS_SAMPLE:
                    image_per_row = 2
                    if self.config.SAMPLE_SIZE <= 6:
                        image_per_row = 1
                    outputs = self.model(
                        images)
                    i = 0
                    for output in outputs:
                        outputs[i] = self.postprocess(output)
                        i += 1
                    matte_gt = GT-images
                    matte_gt = matte_gt - \
                        (matte_gt.min(dim=2, keepdim=True).values).min(
                            dim=3, keepdim=True).values
                    matte_gt = matte_gt / \
                        (matte_gt.max(dim=2, keepdim=True).values).max(
                            dim=3, keepdim=True).values
                    images = stitch_images(
                        self.postprocess(images),
                        outputs,
                        self.postprocess(matte_gt),
                        self.postprocess(GT),
                        img_per_row=image_per_row,
                    )
                    images.save(path)
                else:
                    outputs = self.model(images)
                    outputs = self.postprocess(outputs[-1])[0]
                    path = os.path.join(save_path, name)
                    imsave(outputs, path)
                # debug
                if self.debug:
                    if index == 10:
                        break
        print('\nEnd test....')

    def sample(self, it=None):
        # initial, do not sample when validation set is empty
        if len(self.val_dataset) == 0:
            return
        # torch.cuda.empty_cache()
        self.model.eval()
        iteration = self.model.iteration

        items = next(self.sample_iterator)
        images, mask, GT = self.cuda(
            *items)
        image_per_row = 2
        if self.config.SAMPLE_SIZE <= 6:
            image_per_row = 1
        outputs = self.model(
            images)
        i = 0
        for output in outputs:
            outputs[i] = self.postprocess(output)
            i += 1
        matte_gt = GT-images
        matte_gt = matte_gt - \
            (matte_gt.min(dim=2, keepdim=True).values).min(
                dim=3, keepdim=True).values
        matte_gt = matte_gt / \
            (matte_gt.max(dim=2, keepdim=True).values).max(
                dim=3, keepdim=True).values
        images = stitch_images(
            self.postprocess(images),
            outputs,
            self.postprocess(matte_gt),
            self.postprocess(GT),
            img_per_row=image_per_row,
        )

        path = os.path.join(self.samples_path, self.model_name)
        name = os.path.join(path, "mode_"+str(self.config.MODE) +
                            "_"+str(iteration).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def log(self, logs):
        with open(self.log_file, 'a') as f:
            f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()

    def restart_train_check_lr_scheduler(self):
        checkpoints_path = Path('./checkpoints') / self.config.SUBJECT_WORD
        log_eval_val_ap_path = Path(checkpoints_path) / \
            ('log_eval_val_ap_'+str(self.config.MODE)+'.txt')

        if log_eval_val_ap_path.is_file():
            eval_val_ap = np.genfromtxt(
                log_eval_val_ap_path, dtype=np.str, encoding='utf-8').astype(np.float)
            EPOCH = self.config.EPOCHLIST

            if EPOCH[-1]!=0 and eval_val_ap[0, 1] != EPOCH[EPOCH[-1]-1]-1:
                ap = str(eval_val_ap[0, 0])
                self.scheduler.step(ap)
                if self.config.ADV:
                    for scheduler_dis in self.scheduler_dis:
                        scheduler_dis.step(ap)
