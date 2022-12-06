import numpy as np
import pywt
import matplotlib.pyplot as plt
from wt_function import my_cwt1d,my_dwt2d

file_path = './PTB-XL/ptbxl_resample_cut_z_500.npy' # data path
fs = 500
ptbxl_resample_cut_z_500 = np.load(file_path,allow_pickle=True)
ptbxl_resample_cut_z_500 = np.swapaxes(ptbxl_resample_cut_z_500,1,2)

ptbxl_resample_cut_z_500 = ptbxl_resample_cut_z_500[:100,:,:]

epoch_length = np.shape(ptbxl_resample_cut_z_500)[-1]


ptbxl_resample_cut_z_500_cwt = my_cwt1d(ptbxl_resample_cut_z_500)


ptbxl_resample_cut_z_500_cwt_dwt = my_dwt2d(ptbxl_resample_cut_z_500_cwt)

print('done')

