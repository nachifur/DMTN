import os
import numpy as np

def parpare_image_syn(val_path,stage='train_shadow_free'):

  iminput = val_path
  val_mask_name = val_path.split('/')[-1].split('_')[-1]
  gtmask = val_path.replace(stage,'train_B_ISTD').replace(val_path.split('/')[-1],val_mask_name)

  val_im_name = '_'.join(val_path.split('/')[-1].split('_')[0:-1])+'.jpg'
  imtarget = val_path.replace(stage,'shadow_free').replace(val_path.split('/')[-1],val_im_name)

  return iminput,imtarget,gtmask

def prepare_data(train_path, stage=['train_A']):
    input_names=[]
    for dirname in train_path:
        for subfolder in stage:
            train_b = dirname + "/"+ subfolder+"/"
            for root, _, fnames in sorted(os.walk(train_b)):
                for fname in fnames:
                    if is_image_file(fname):
                        input_names.append(os.path.join(train_b, fname))
    return input_names

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def gen_syn_images_flist(train_real_root):
    train_real_root = [train_real_root]
    syn_images=prepare_data(train_real_root,stage=['synC'])

    shadow_syn_list= []
    shadow_free_syn_list= []
    mask_syn_list= []
    for i in range(len(syn_images)):
        shadow,shadow_free,mask = parpare_image_syn(syn_images[i],stage='synC')
        shadow_syn_list.append(shadow)
        shadow_free_syn_list.append(shadow_free)
        mask_syn_list.append(mask)
    return shadow_syn_list,shadow_free_syn_list,mask_syn_list

