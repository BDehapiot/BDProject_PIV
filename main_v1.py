#%% Imports

import time
import napari
import numpy as np
from skimage import io 
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.signal import correlate
from skimage.transform import rescale

#%% Open data

# -----------------------------------------------------------------------------

stack_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8.tif'
# stack_name = 'Xenopus-Cilia_250fps_Mesdjian_uint8_lite.tif'

# -----------------------------------------------------------------------------

mask_name = '18-07-11_40x_GBE_UtrCH_Ctrl_b1_uint8_maskProj.tif'

# -----------------------------------------------------------------------------

stack = io.imread(Path('data', stack_name))
if 'mask_name' in locals():
    mask = io.imread(Path('data', mask_name))
else:
    mask = None

#%% Parameters

intSize = 32
srcSize = 64
binning = 1

#%% Initialize

# Rescale data
stack = rescale(stack, (1, 1/binning, 1/binning), preserve_range=True)
intSize = intSize//binning
srcSize = srcSize//binning

# Get stack shape variables
nT = stack.shape[0]
nY = stack.shape[1]
nX = stack.shape[2]

# Get number of interrogation windows
nIntY = nY//intSize
nIntX = nX//intSize

# Pad stack to fit search windows
pad = srcSize//2
stackPad = np.pad(stack, pad_width=((0, 0), (pad, pad), (pad, pad)))

# Define interrogation window origins
intYi = np.arange(
    (nY-nIntY*intSize)//2, 
    (nY-nIntY*intSize)//2 + nIntY*intSize, 
    intSize
    ) + pad
intXi = np.arange(
    (nX-nIntX*intSize)//2, 
    (nX-nIntX*intSize)//2 + nIntX*intSize, 
    intSize
    ) + pad

# Define search window origins
srcYi = intYi-(srcSize-intSize)//2
srcXi = intXi-(srcSize-intSize)//2 

#%% Compute vector field

v = np.full((nT, nIntY, nIntX), np.nan)
u = np.full((nT, nIntY, nIntX), np.nan)

start = time.time()
print('2D correlation')

for t in range(1, nT):       
    for y in range(nIntY):
        for x in range(nIntX):
            
            # Extract interrogation window
            intWin = stackPad[
                t-1,
                intYi[y]:intYi[y]+intSize,
                intXi[x]:intXi[x]+intSize,
                ]

            # Extract search window
            srcWin = stackPad[
                t,
                srcYi[y]:srcYi[y]+srcSize,
                srcXi[x]:srcXi[x]+srcSize,
                ]

            # Compute 2D correlation
            corr2D = correlate(
                srcWin - np.mean(srcWin), 
                intWin - np.mean(intWin),
                method='fft',
                )
            
            # Find xy max correlation
            y_max, x_max = np.unravel_index(corr2D.argmax(), corr2D.shape)            
            
            # Infer uv vector components
            u[t,y,x] = x_max - (intSize-1) - (srcSize//2-intSize//2)
            v[t,y,x] = y_max - (intSize-1) - (srcSize//2-intSize//2)

end = time.time()
print(f'  {(end-start):5.3f} s') 

#%% Filter vector field



#%% Display (matplotlib)

fig, ax = plt.subplots(figsize=(6, 6))
ax.quiver(u[1,...], v[1,...])


#%% Display (napari)

# viewer = napari.Viewer()
# viewer.add_image(corr2D)
# viewer.add_image(v)
# viewer.add_image(u)
