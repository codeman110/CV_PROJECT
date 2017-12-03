########################################
# Final algorithm for retinal videos #
########################################

import imageio
import numpy as np
import cv2

# Functions
def build_gdown_stack(vid, g_ht, g_wd, n_frames, level):
    result = np.zeros((n_frames,g_ht,g_wd,3),dtype='float')
    for i in range(0,n_frames):
        # generate Gaussian pyramid
        G = vid[i].copy()
        for j in xrange(level):
            G = cv2.pyrDown(G)
        result[i] = G
    return result

def ideal_bandpassing(ip, wl, wh, g_ht, g_wd, n_frames, samplingRate):
    n = n_frames - 10
    Dimensions = np.array([1,g_ht,g_wd,3])    
    Freq = np.arange(1,n+1,dtype='float')
    Freq = (Freq-1)/n*samplingRate 
    mask = np.zeros((n_frames),dtype='b')
    for i in range(len(Freq)):
        if(wl < Freq[i] < wh):
            mask[i] = 1
    new_mask = np.tile(mask[:,None,None,None],Dimensions)

    # Alternative approach
#    new_mask = np.zeros((n_frames,g_ht,g_wd,3))
#    new_mask[9,:,:,:] = 1
    
    F = np.fft.fft(ip,axis=0) 
    F = np.multiply(F,new_mask)    
    filtered = np.real(np.fft.ifft(F,axis=0))   
    return filtered

def compute_downsampled_params(width,height,pyr_lvls):
    g_ht = height
    g_wd = width
    for i in range(pyr_lvls):
        g_ht = round(g_ht/2)
        g_wd = round(g_wd/2)
    return int(g_ht),int(g_wd)

# Parameters

file_raw = 'data/out_raw.mp4'    # video file path
file_mask = 'data/out_mask.mp4'

# Reading video file
reader1 = imageio.get_reader(file_raw,  'ffmpeg')
reader2 = imageio.get_reader(file_mask,  'ffmpeg')

alpha = 5                 # amplification factor
w_l = 0.5 #50/60.0              # lower_hertz (in Hz)
w_h = 10 #60/60.0              # upper_hertz (in Hz)
sampling_rate = 30         # sampling rate
pyr_lvls = 4               # nmbr of pyramid levels
n_frames = reader1.get_meta_data()['nframes']
height = 578
width = 524
fps = 8

# Algorithm

# Computing height and width of downsampled frames
g_ht,g_wd = compute_downsampled_params(width,height,pyr_lvls)

# Initializing 4D numpy arrays with dimensions frames, height, width and channels
im_mas = np.zeros((n_frames, height, width, 3), dtype='uint8')
im_raw = np.zeros((n_frames, height, width, 3), dtype='uint8')
im = np.zeros((n_frames, height, width, 3), dtype='uint8')
stack = np.zeros((n_frames, height, width, 3), dtype='float')



# Copying video data to 4D array
for i,frame in enumerate(reader1):
    im_raw[i] = frame
    
for i,frame in enumerate(reader2):
    im_mas[i] = frame

im_cp = im_raw.copy()
im_cp[im_cp < 100] = 0
im_cp[:,:,:,1] = 0
im_cp[:,:,:,2] = 0


#imageio.imwrite('sample.png',a)
#cv2.imshow('x',a)

# Compute Gaussian blur stack
gauss_stack = build_gdown_stack(im_cp, g_ht, g_wd, n_frames, pyr_lvls)

# Temporal filtering
filtered_stack = ideal_bandpassing(gauss_stack, w_l, w_h, g_ht, g_wd, n_frames, sampling_rate)

# Amplify
filtered_stack = np.multiply(filtered_stack,alpha)

# Resizing filtered_stack
for i in range(0,n_frames):
    stack[i] = cv2.resize(filtered_stack[i], (width, height), interpolation = cv2.INTER_CUBIC)

# Adding the original frames and filtered frames
vid = np.add(im,stack)
vid = im_cp + vid       # adding amplified image to original
vid[vid>255] = 255
vid[vid<0] = 0
vid = vid.astype(dtype='uint8')

#vid = cv2.add(im_m,vid)


im1 = cv2.bitwise_or(vid,im_mas)    # discarding amplified background

im2 = cv2.bitwise_or(im_raw,cv2.bitwise_not(im_mas))    # taking the background from raw images

im3 = cv2.bitwise_and(im1,im2)  # adding the foreground and the background

#Creating output video file
writer = imageio.get_writer('op_final.mp4', fps=fps, macro_block_size=None)
for i,frame in enumerate(im3):
    writer.append_data(frame)
writer.close()

