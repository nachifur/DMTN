import os
import numpy as np
from pathlib import Path

def gen_flist(data_path):
    ext = {'.JPG', '.JPEG', '.PNG', '.TIF', 'TIFF','json'}
    images = []
    for root, dirs, files in os.walk(data_path):
        print('loading ' + root)
        for file in files:
            if os.path.splitext(file)[1].upper() in ext:
                images.append(os.path.join(root, file))

    images = sorted(images)
    return images
