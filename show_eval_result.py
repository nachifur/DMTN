import numpy as np
rmse = np.genfromtxt(
    'checkpoints/ISTD/show_result.txt', dtype=np.str, encoding='utf-8').astype(np.float)

print('running rmse-shadow: %.4f, rmse-non-shadow: %.4f, rmse-all: %.4f'
        % (rmse[0], rmse[1], rmse[2]))
