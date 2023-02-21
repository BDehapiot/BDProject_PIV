#%% Imports

import cv2
import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import correlate
from skimage.transform import rescale

#%% Comments

'''
- add subpixel max 2Dcorr detection?
'''

#%% Open data

# -----------------------------------------------------------------------------

# stack_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8.tif'
stack_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_lite.tif'
# stack_name = 'Xenopus-Cilia_250fps_Mesdjian_uint8_lite.tif'

# -----------------------------------------------------------------------------

mask_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_maskProj.tif'
# mask_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_maskAll.tif'

# -----------------------------------------------------------------------------

stack = io.imread(Path('data', stack_name))
if 'mask_name' in locals():
    mask = io.imread(Path('data', mask_name))
else:
    mask = None

#%% Parameters

intSize = 32 # size of interrogation window (pixels)
srcSize = 64 # size of search window (pixels)
binning = 2 # reduce image size to speed up computation (1, 2, 4, 8...)
maskCutOff = 1 # mask out interrogation windows (???) 

#%% Initialize

# Adjust parameters acc. to binning
intSize = intSize//binning
srcSize = srcSize//binning
srcPad = (srcSize-intSize)//2

# Rescale data acc. to binning
stack = rescale(stack, (1, 1/binning, 1/binning), preserve_range=True)
if mask is not None:
    mask = mask.astype('bool')
    if mask.ndim == 2:
        mask = rescale(mask, (1/binning, 1/binning), order=0)
    if mask.ndim == 3: 
        mask = rescale(mask, (1, 1/binning, 1/binning), order=0)

# Count number of int. window
intYn = (stack.shape[1]-srcPad*2)//intSize
intXn = (stack.shape[2]-srcPad*2)//intSize

# Setup int. & src window coordinates
intYi = np.arange(
    (stack.shape[1]-intYn*intSize)//2, 
    (stack.shape[1]-intYn*intSize)//2 + intYn*intSize, 
    intSize,
    )
intXi = np.arange(
    (stack.shape[2]-intXn*intSize)//2, 
    (stack.shape[2]-intXn*intSize)//2 + intXn*intSize, 
    intSize,
    )
srcYi = intYi - srcPad
srcXi = intXi - srcPad 

#%%

stack_test = np.zeros_like(stack[0,...])
      
for y in range(intYn):
    for x in range(intXn):
        
        stack_test[
            srcYi[y]:srcYi[y]+srcSize, 
            srcXi[x]:srcXi[x]+srcSize,
            ] = 1  
                
for y in range(intYn):
    for x in range(intXn):
                
        stack_test[
            intYi[y]:intYi[y]+intSize, 
            intXi[x]:intXi[x]+intSize,
            ] = 2
        
        stack_test[intYi[y],:] = 0
        stack_test[:,intXi[x]] = 0
        
#%% Display (napari)

viewer = napari.Viewer()
viewer.add_image(stack_test)
            
            
