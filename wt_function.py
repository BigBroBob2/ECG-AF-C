import numpy as np
import pywt

def my_cwt1d(ptbxl,time_axis=-1,fs=500,scales=np.arange(1,65),wavelet='gaus8',Repeat=10):
    """1-D CWT

    Args:
        ptbxl (array_like): batch*channel*time
        time_axis (int, optional): time axis. Defaults to -1.
        fs (int, optional): sampling rate. Defaults to 500.
        scales (array_like, optional): wavelet scales. Defaults to np.arange(1,65).
        wavelet (str, optional): see more from pywt.wavelist(kind='continuous'), ['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']. Defaults to 'gaus8'.
        Repeat (int, optional): number of repetition to avoid edge effect. Defaults to 10.

    Returns:
        array_like: batch*channel*time*scale
    """    
    epoch_length = np.shape(ptbxl)[time_axis]
    
    prod = np.ones(len(np.shape(ptbxl)))
    prod[time_axis] = Repeat
    prod = tuple(prod.astype(int))

    ptbxl_repeat = pywt.cwt(np.kron(np.ones(prod), ptbxl),scales,wavelet,axis=time_axis)[0]
    ptbxl_cwt1 = np.take(ptbxl_repeat,indices=np.arange(epoch_length,2*epoch_length),axis=time_axis)
    for r in range(2,Repeat-1):
        ptbxl_cwt1 = ptbxl_cwt1*(1.0*(r-1)/r) + np.take(ptbxl_repeat,indices=np.arange(r*epoch_length,(r+1)*epoch_length),axis=time_axis)*(1.0/(r))
    ptbxl_cwt1 = np.transpose(ptbxl_cwt1,(1,2,3,0))
    # freq = pywt.scale2frequency(wavelet,scales) / (epoch_length/fs)
    return ptbxl_cwt1

def my_dwt2d(ptbxl_cwt,time_axis=-2,scale_axis=-1,wavelet='db4',mode='reflect'):
    """1-Level 2-D DWT

    Args:
        ptbxl_cwt (array_like): batch*channel*time*scale
        time_axis (int, optional): time axis. Defaults to -2.
        scale_axis (int, optional): scale axis. Defaults to -1.
        wavelet (str, optional): see more from pywt.wavelist(kind="discrete"), ['haar', 'db', 'sym', 'coif', 'bior', 'rbio', 'dmey', 'gaus', 'mexh', 'morl', 'cgau', 'shan', 'fbsp', 'cmor']. Defaults to 'db4'.
        mode (str, optional): ['zero', 'constant', 'symmetric', 'periodic', 'smooth', 'periodization', 'reflect', 'antisymmetric', 'antireflect']. Defaults to 'reflect'.

    Returns:
        _type_: _description_
    """    
    # max_level = pywt.dwtn_max_level(np.shape(ptbxl_cwt),wavelet,axes=[time_axis,scale_axis])
    ptbxl_cwt_dwt = []
    ptbxl_cwt_dwt.append(pywt.dwtn(ptbxl_cwt,wavelet,mode,axes=[time_axis,scale_axis]))
    ptbxl_cwt_dwt = np.concatenate([ptbxl_cwt_dwt[-1]['aa'][:,:,:,:,np.newaxis],
                                    ptbxl_cwt_dwt[-1]['ad'][:,:,:,:,np.newaxis],
                                    ptbxl_cwt_dwt[-1]['da'][:,:,:,:,np.newaxis],
                                    ptbxl_cwt_dwt[-1]['dd'][:,:,:,:,np.newaxis]],axis=-1)
    return ptbxl_cwt_dwt