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
binning = 2 # bin data to speed up computation (1, 2, 4, 8...)
maskCutOff = 1

#%% Initialize

# Adjust parameters acc. to binning
intSize = intSize//binning
srcSize = srcSize//binning
pad = (srcSize-intSize)//2

# Rescale data acc. to binning
stack = rescale(stack, (1, 1/binning, 1/binning), preserve_range=True)
if mask is not None:
    mask = mask.astype('bool')
    if mask.ndim == 2:
        mask = rescale(mask, (1/binning, 1/binning), order=0)
    if mask.ndim == 3: 
        mask = rescale(mask, (1, 1/binning, 1/binning), order=0)

# Count number of interrogation window
intYn = stack.shape[1]//intSize
intXn = stack.shape[2]//intSize

# Setup interrogation and search windows
intYi = np.arange(
    (stack.shape[1]-intYn*intSize)//2, 
    (stack.shape[1]-intYn*intSize)//2 + intYn*intSize, 
    intSize
    ) + pad
intXi = np.arange(
    (stack.shape[2]-intXn*intSize)//2, 
    (stack.shape[2]-intXn*intSize)//2 + intXn*intSize, 
    intSize
    ) + pad
srcYi = intYi - pad
srcXi = intXi - pad 

# Pad data
stack = np.pad(stack, pad_width=((0, 0), (pad, pad), (pad, pad)))
if mask is not None:
    mask = mask.astype('bool')
    if mask.ndim == 2:
        mask = np.pad(mask, pad_width=pad) 
    if mask.ndim == 3: 
        mask = np.pad(mask, pad_width=((0, 0), (pad, pad), (pad, pad))) 

#%% Compute vector field

v = np.full((stack.shape[0], intYn, intXn), np.nan)
u = np.full((stack.shape[0], intYn, intXn), np.nan)
sd = np.full((stack.shape[0], intYn, intXn), np.nan)

intWin_all = []
srcWin_all = []

start = time.time()
print('2D correlation')

for t in range(1, stack.shape[0]):       
    for y in range(intYn):
        for x in range(intXn):
            
            # Extract mask interrogation window 
            if mask is not None:
                if mask.ndim == 2:
                    maskWin = mask[
                        intYi[y]:intYi[y]+intSize,
                        intXi[x]:intXi[x]+intSize,                        
                        ]
                if mask.ndim == 3:
                    maskWin = mask[t-1,
                        intYi[y]:intYi[y]+intSize,
                        intXi[x]:intXi[x]+intSize,                        
                        ]              
                
            if mask is None or np.mean(maskWin) >= maskCutOff:
                                
                # Extract interrogation & search window
                intWin = stack[t-1,
                    intYi[y]:intYi[y]+intSize, 
                    intXi[x]:intXi[x]+intSize,
                    ]
                srcWin = stack[t,
                    srcYi[y]:srcYi[y]+srcSize, 
                    srcXi[x]:srcXi[x]+srcSize,
                    ]              
    
                # Compute 2D correlation
                corr2D = correlate(
                    srcWin - np.mean(srcWin), 
                    intWin - np.mean(intWin),
                    method='fft',
                    )
                                
                intWin_all.append(intWin)
                srcWin_all.append(srcWin)  
                
                # Find xy max correlation
                y_max, x_max = np.unravel_index(corr2D.argmax(), corr2D.shape)            
                
                # Infer uv vector components
                u[t,y,x] = x_max - (intSize-1) - (srcSize//2-intSize//2)
                v[t,y,x] = y_max - (intSize-1) - (srcSize//2-intSize//2)
                
            else:
                
                u[t,y,x] = np.nan
                v[t,y,x] = np.nan
                
            # # Evaluate peak signal to noise ratio (PSNR)            
            # sd[t,y,x] = np.std(
            #     stack[t-1,
            #         intYi[y]:intYi[y]+intSize,
            #         intXi[x]:intXi[x]+intSize,
            #         ]
            #     )

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%% Filter vector field


#%% Display (matplotlib)

fig, ax = plt.subplots(figsize=(6, 6))
ax.quiver(u[1,...], v[1,...])


#%% Display (napari)

viewer = napari.Viewer()
# viewer.add_image(v)
# viewer.add_image(u)
# viewer.add_image(sd)
# viewer.add_image(np.swapaxes(np.dstack(intWin_all), 2, 0))
viewer.add_image(np.swapaxes(np.dstack(srcWin_all), 2, 0))
