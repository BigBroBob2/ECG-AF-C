import numpy as np
import pywt
import matplotlib.pyplot as plt

file_path = './ptbxl_resample_cut_z_500.npy'
fs = 500
ptbxl_resample_cut_z_500 = np.load(file_path,allow_pickle=True)
epoch_length = np.shape(ptbxl_resample_cut_z_500)[1]

'''
pywt.cwt
wavelet:
pywt.wavelist(kind='continuous')
['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']
'''
wavelet_cwt = 'gaus8'
scales = np.arange(1,65)
Repeat = 10 # should > 2
ptbxl_resample_cut_z_500_cwt1_repeat = pywt.cwt(np.kron(np.ones((1,Repeat,1)), ptbxl_resample_cut_z_500[:100,:,:]),scales,wavelet_cwt,axis=1)[0]

ptbxl_resample_cut_z_500_cwt1 = ptbxl_resample_cut_z_500_cwt1_repeat[:,:,epoch_length:2*epoch_length,:]
for r in range(2,Repeat-1):
    ptbxl_resample_cut_z_500_cwt1 = ptbxl_resample_cut_z_500_cwt1*(1.0*(r-1)/r) + ptbxl_resample_cut_z_500_cwt1_repeat[:,:,r*epoch_length:(r+1)*epoch_length,:]*(1.0/(r))
# plt.imshow(ptbxl_resample_cut_z_500_cwt1[:,0,:,0])
freq = pywt.scale2frequency(wavelet_cwt,scales) / (epoch_length/fs)

'''
pywt.dwt2
wavelet:
['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']
    * Haar (``haar``)
    * Daubechies (``db``)
    * Symlets (``sym``)
    * Coiflets (``coif``)
    * Biorthogonal (``bior``)
    * Reverse biorthogonal (``rbio``)
    * `"Discrete"` FIR approximation of Meyer wavelet (``dmey``)
    * Gaussian wavelets (``gaus``)
    * Mexican hat wavelet (``mexh``)
    * Morlet wavelet (``morl``)
    * Complex Gaussian wavelets (``cgau``)
    * Shannon wavelets (``shan``)
    * Frequency B-Spline wavelets (``fbsp``)
    * Complex Morlet wavelets (``cmor``)

mode:
['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect']
    zero - zero-padding                   0  0 | x1 x2 ... xn | 0  0
    constant - constant-padding          x1 x1 | x1 x2 ... xn | xn xn
    symmetric - symmetric-padding        x2 x1 | x1 x2 ... xn | xn xn-1
    reflect - reflect-padding            x3 x2 | x1 x2 ... xn | xn-1 xn-2
    periodic - periodic-padding        xn-1 xn | x1 x2 ... xn | x1 x2
    smooth - smooth-padding             (1st derivative interpolation)
    antisymmetric -                    -x2 -x1 | x1 x2 ... xn | -xn -xn-1
    antireflect -                      -x3 -x2 | x1 x2 ... xn | -xn-1 -xn-2
    periodization - like periodic-padding but gives the smallest possible
                    number of decomposition coefficients. IDWT must be
                    performed with the same mode.

'''
wavelet_dwt = 'db4'
mode = 'reflect'
max_level = pywt.dwtn_max_level(np.shape(ptbxl_resample_cut_z_500_cwt1),wavelet_dwt,axes=[0,2])

ptbxl_resample_cut_z_500_cwt1_dwt2 = []
temp = ptbxl_resample_cut_z_500_cwt1
for i in range(max_level):
    ptbxl_resample_cut_z_500_cwt1_dwt2.append(pywt.dwtn(temp,wavelet_dwt,mode,axes=[0,2]))
    temp = ptbxl_resample_cut_z_500_cwt1_dwt2[-1]['aa']
# plt.imshow(ptbxl_resample_cut_z_500_cwt1_dwt2[-1]['aa'][:,0,:,0])
print('done')

