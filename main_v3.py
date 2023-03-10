#%% Imports

import time
import numpy as np
from skimage import io 
from pathlib import Path
from joblib import Parallel, delayed 

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

# mask_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_maskProj.tif'
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
maskCutOff = 0 # mask out interrogation windows (???) 
parallel = True

#%% Initialize

from skimage.transform import rescale

# -----------------------------------------------------------------------------

# Adjust parameters acc. to binning
intSize = intSize//binning
srcSize = srcSize//binning 
if intSize % 2 != 0:
    intSize -= intSize % 2
if srcSize % 2 != 0:
    srcSize -= srcSize % 2
srcPad = (srcSize-intSize)//2
print(f'intSize = {intSize*binning}')
print(f'srcSize = {srcSize*binning}')

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

# Setup int. & src. window coordinates
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

#%% Compute vector field

from scipy.signal import correlate

# -----------------------------------------------------------------------------

# Create empty arrays
v = np.full((stack.shape[0], intYn, intXn), np.nan)
u = np.full((stack.shape[0], intYn, intXn), np.nan)

start = time.time()
print('2D correlation')

for t in range(1, stack.shape[0]):       
    for y, (iYi, sYi) in enumerate(zip(intYi, srcYi)):
        for x, (iXi, sXi) in enumerate(zip(intXi, srcXi)):
            
            # Extract mask int. window 
            if mask is not None:
                if mask.ndim == 2:
                    maskWin = mask[iYi:iYi+intSize,iXi:iXi+intSize]
                if mask.ndim == 3:
                    maskWin = mask[t-1,iYi:iYi+intSize,iXi:iXi+intSize]
            
            if mask is None or np.mean(maskWin) >= maskCutOff:
            
                # Extract int. & src. window
                intWin = stack[t-1,iYi:iYi+intSize,iXi:iXi+intSize]
                srcWin = stack[t,sYi:sYi+srcSize,sXi:sXi+srcSize]              
    
                # Compute 2D correlation
                corr2D = correlate(
                    srcWin - np.mean(srcWin), 
                    intWin - np.mean(intWin),
                    method='fft'
                    )
                
                # Find max corr. and infer uv components
                y_max, x_max = np.unravel_index(corr2D.argmax(), corr2D.shape)            
                u[t,y,x] = x_max-(intSize-1)-(srcSize//2-intSize//2)
                v[t,y,x] = y_max-(intSize-1)-(srcSize//2-intSize//2)
                
            else:
                
                u[t,y,x] = np.nan
                v[t,y,x] = np.nan
            
end = time.time()
print(f'  {(end-start):5.3f} s') 

#%% Display (vector field)

# import matplotlib.pyplot as plt

# # -----------------------------------------------------------------------------

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.quiver(u[1,...], v[1,...])


#%% Display (images)

# import napari

# -----------------------------------------------------------------------------

# viewer = napari.Viewer()
# viewer.add_image(roiDisplay)

#%%

# from skimage.draw import rectangle

# # -----------------------------------------------------------------------------

# roiDisplay = stack[stack.shape[0]//2,...].copy()
# roiDisplay = np.repeat(roiDisplay[None,:], intYn*intXn, axis=0)
      
# nWin = 0
# maxDisplay = np.max(roiDisplay)
# for iYi, sYi in zip(intYi, srcYi):
#     for iXi, sXi in zip(intXi, srcXi):
        
#         # Print srcWin
#         start = (sYi, sXi)
#         extent = (srcSize, srcSize)
#         rr, cc = rectangle(
#             start, 
#             extent=extent, 
#             shape=roiDisplay[0,...].shape
#             )
#         roiDisplay[nWin, rr, cc] = maxDisplay/2   
        
#         # Print intWin
#         start = (iYi, iXi)
#         extent = (intSize, intSize)
#         rr, cc = rectangle(
#             start, 
#             extent=extent, 
#             shape=roiDisplay[0,...].shape
#             )
#         roiDisplay[nWin, rr, cc] = maxDisplay
        
#         nWin += 1
            
            
