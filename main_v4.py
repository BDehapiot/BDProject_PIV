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
# stack_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_lite.tif'
# stack_name = 'Xenopus-Cilia_250fps_Mesdjian_uint8_lite.tif'
stack_name = '18-07-03_100x_UtrCH_Ctrl_a2_uint8.tif'

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

intSize = 24 # size of interrogation window (pixels)
srcSize = 48 # size of search window (pixels)
binning = 2 # reduce image size to speed up computation (1, 2, 4, 8...)
maskCutOff = 1 # mask out interrogation windows (???) 
parallel = True

#%% Functions

from scipy.signal import correlate
from skimage.transform import rescale

# -----------------------------------------------------------------------------

def getPIV(
        stack,
        intSize=32,
        srcSize=64,
        binning=1,
        mask=None,
        maskCutOff=1,
        parallel=True
        ):
    
    # Nested function ---------------------------------------------------------
    
    def _getPIV(img, ref):
        
        # Create empty arrays
        vecU = np.full((intYn, intXn), np.nan)
        vecV = np.full((intYn, intXn), np.nan)
        
        for y, (iYi, sYi) in enumerate(zip(intYi, srcYi)):
            for x, (iXi, sXi) in enumerate(zip(intXi, srcXi)):
                
                # Extract mask int. window 
                if mask is not None:
                    if mask.ndim == 2:
                        maskWin = mask[iYi:iYi+intSize,iXi:iXi+intSize]
                    if mask.ndim == 3:
                        maskWin = mask[iYi:iYi+intSize,iXi:iXi+intSize]
                
                if mask is None or np.mean(maskWin) >= maskCutOff:
                
                    # Extract int. & src. window
                    intWin = ref[iYi:iYi+intSize,iXi:iXi+intSize]
                    srcWin = img[sYi:sYi+srcSize,sXi:sXi+srcSize]           
        
                    # Compute 2D correlation
                    corr2D = correlate(
                        srcWin - np.mean(srcWin), 
                        intWin - np.mean(intWin),
                        method='fft'
                        )
                    
                    # Find max corr. and infer uv components
                    y_max, x_max = np.unravel_index(corr2D.argmax(), corr2D.shape)            
                    vecU[y,x] = x_max-(intSize-1)-(srcSize//2-intSize//2)
                    vecV[y,x] = y_max-(intSize-1)-(srcSize//2-intSize//2)
                    
                else:
                    
                    vecU[y,x] = np.nan
                    vecV[y,x] = np.nan
        
        return vecU, vecV
        
    # Run ---------------------------------------------------------------------

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

    # _getPIV
    if parallel:
        
        output_list = Parallel(n_jobs=-1)(
        delayed(_getPIV)(
            stack[t,...],
            stack[t-1,...]
            )
        for t in range(1, stack.shape[0])
        )
        
    else:
        
        output_list = [_getPIV(
            stack[t,...],
            stack[t-1,...]
            )
        for t in range(1, stack.shape[0])
        ]
        
    # Fill output dictionary    
    output_dict = {
    
    # Parameters
    'intSize': intSize,
    'srcSize': srcSize,
    'binning': binning,
    'maskCutOff': maskCutOff,
    
    # Data
    'intYi': intYi,
    'intXi': intXi,
    'vecU': np.stack(
        [data[0] for data in output_list], axis=0),
    'vecV': np.stack(
        [data[1] for data in output_list], axis=0),

    }
        
    return output_dict

start = time.time()
print('2D correlation')
        
output_dict = getPIV(
    stack,
    intSize=intSize,
    srcSize=srcSize,
    binning=binning,
    mask=mask,
    maskCutOff=maskCutOff,
    parallel=True
    )

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%% Filter results 

from scipy.stats import zscore
from bdtools.nan import nanreplace, nanfilt

# -----------------------------------------------------------------------------
   
# Extract parameters & data
vecU = output_dict['vecU']
vecV = output_dict['vecV']

# 

# outTresh = 1.5
    
# for t, (u, v) in enumerate(zip(vecU, vecV)):
    
#     temp_mask = ~np.isnan(u)
#     norm = np.hypot(u, v)
#     z_u = np.abs(zscore(u, axis=None, nan_policy='omit'))
#     z_v = np.abs(zscore(v, axis=None, nan_policy='omit'))
#     u[(z_u>outTresh) | (z_v>outTresh)] = np.nan
#     v[(z_u>outTresh) | (z_v>outTresh)] = np.nan
    
#     u = nanreplace(
#         u, 
#         kernel_size=3, 
#         method='mean', 
#         mask=temp_mask,
#         )

#     v = nanreplace(
#         v, 
#         kernel_size=3, 
#         method='mean', 
#         mask=temp_mask,
#         )
    
#     vecU[t,...] = nanfilt(
#         u, 
#         kernel_size=3, 
#         method='mean', 
#         iterations=3,
#         )

#     vecV[t,...] = nanfilt(
#         v, 
#         kernel_size=3, 
#         method='mean', 
#         iterations=3,
#         )   

#%% Display (vector field)

# import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

# t = 40
# u = vecU[t,...]
# v = vecV[t,...]
# norm = np.hypot(u, v)
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.quiver(u, v, norm)

#%% Display (images)

# import napari

# -----------------------------------------------------------------------------

# viewer = napari.Viewer()
# viewer.add_image(roiDisplay)           
