import os
import numpy as np
from pathlib import Path
from flist import gen_flist
from flist_train_val_test import gen_flist_train_val_test
from process_syn_data import gen_syn_images_flist

if __name__ == '__main__':
    # ISTD
    ISTD_path = "/gpfs/home/liujiawei/data-set/shadow_removal_dataset/ISTD_Dataset_arg"

    save_path = ISTD_path+"/data_val"

    if not Path(save_path).exists():
        os.mkdir(save_path)

    seed = 10

    # sys image 
    shadow_syn_list,shadow_free_syn_list,mask_syn_list = gen_syn_images_flist(ISTD_path+'/train')

    # train and val (data augmentation)
    data_path = Path(ISTD_path)/"train/train_A"
    flist_main_name = "ISTD_shadow.flist"
    flist_save_path = Path(save_path)/flist_main_name
    images = gen_flist(data_path)
    images+=shadow_syn_list
    np.savetxt(flist_save_path, images, fmt='%s')
    png_val_test_PATH = [Path(save_path)/str(Path(flist_save_path).stem+'_pre_train.flist'),
                         Path(save_path)/str(Path(flist_save_path).stem+'_pre_val.flist'), ""]
    id_list = gen_flist_train_val_test(
        flist_save_path, png_val_test_PATH, [8, 2, 0], seed, [])

    data_path = Path(ISTD_path)/"train/train_B"
    flist_main_name = "ISTD_mask.flist"
    flist_save_path = Path(save_path)/flist_main_name
    images = gen_flist(data_path)
    images+=mask_syn_list
    np.savetxt(flist_save_path, images, fmt='%s')
    png_val_test_PATH = [Path(save_path)/str(Path(flist_save_path).stem+'_pre_train.flist'),
                         Path(save_path)/str(Path(flist_save_path).stem+'_pre_val.flist'), ""]
    gen_flist_train_val_test(flist_save_path, png_val_test_PATH, [
                             8, 2, 0], seed, id_list)

    data_path = Path(ISTD_path)/"train/train_C"
    flist_main_name = "ISTD_shadow_free.flist"
    flist_save_path = Path(save_path)/flist_main_name
    images = gen_flist(data_path)
    images+=shadow_free_syn_list
    np.savetxt(flist_save_path, images, fmt='%s')
    png_val_test_PATH = [Path(save_path)/str(Path(flist_save_path).stem+'_pre_train.flist'),
                         Path(save_path)/str(Path(flist_save_path).stem+'_pre_val.flist'), ""]
    gen_flist_train_val_test(flist_save_path, png_val_test_PATH, [
                             8, 2, 0], seed, id_list)

    # train and val
    data_path = Path(ISTD_path)/"train/train_A"
    flist_save_path = Path(save_path)/"ISTD_shadow_train.flist"
    images = gen_flist(data_path)
    np.savetxt(flist_save_path, images, fmt='%s')
    png_val_test_PATH = [Path(save_path)/str(Path(flist_save_path).stem+'_train.flist'),
                         Path(save_path)/str(Path(flist_save_path).stem+'_val.flist'), ""]
    id_list = gen_flist_train_val_test(
        flist_save_path, png_val_test_PATH, [8, 2, 0], seed, [])

    data_path = Path(ISTD_path)/"train/train_B"
    flist_save_path = Path(save_path)/"ISTD_mask_train.flist"
    images = gen_flist(data_path)
    np.savetxt(flist_save_path, images, fmt='%s')
    png_val_test_PATH = [Path(save_path)/str(Path(flist_save_path).stem+'_train.flist'),
                         Path(save_path)/str(Path(flist_save_path).stem+'_val.flist'), ""]
    gen_flist_train_val_test(flist_save_path, png_val_test_PATH, [
                             8, 2, 0], seed, id_list)

    data_path = Path(ISTD_path)/"train/train_C"
    flist_save_path = Path(save_path)/"ISTD_shadow_free_train.flist"
    images = gen_flist(data_path)
    np.savetxt(flist_save_path, images, fmt='%s')
    png_val_test_PATH = [Path(save_path)/str(Path(flist_save_path).stem+'_train.flist'),
                         Path(save_path)/str(Path(flist_save_path).stem+'_val.flist'), ""]
    gen_flist_train_val_test(flist_save_path, png_val_test_PATH, [
                             8, 2, 0], seed, id_list)

    # test
    data_path = Path(ISTD_path)/"test/test_A"
    flist_save_path = Path(save_path)/"ISTD_shadow_test.flist"
    images = gen_flist(data_path)
    np.savetxt(flist_save_path, images, fmt='%s')

    data_path = Path(ISTD_path)/"test/test_B"
    flist_save_path = Path(save_path)/"ISTD_mask_test.flist"
    images = gen_flist(data_path)
    np.savetxt(flist_save_path, images, fmt='%s')

    data_path = Path(ISTD_path)/"test/test_C"
    flist_save_path = Path(save_path)/"ISTD_shadow_free_test.flist"
    images = gen_flist(data_path)
    np.savetxt(flist_save_path, images, fmt='%s')

