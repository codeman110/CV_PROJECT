# INCOMPLETE
# COMPUTER VISION PROJECT
## RETINAL BLOOD FLOW MAGNIFIATION USING EULERIAN VIDEO MAGNIFICATION
The project presents the application of Eulerian Video Magnification on retinal video angiographic images.
## Libraries
- imageio
- opencv
- numpy
## Steps
1. Preprocess fluorescein angiogram (FA) images. Preprocessing involves proper alignment of images and image segmentation to separate blood vessels from background. Preprocessimg steps and data are not shown in this project. Separately mask images are also created for respective FA images.
2. The preprocessed images are then converted to video. In our example we had 54 FA images. The output video has a framerate of 3 fps and a duration of 18 secs. Similarly a mask video with same properties is also created. These are saved as ***op_raw.mp4*** and ***op_masks.mp4*** respectively in the data folder.
3. The above FA videos are further motion interpolated using [butterflow algorithm](https://github.com/dthpham/butterflow) to get smooth motion. Using ```butterflow -r 30 -s a=0,b=end,spd=0.25 <video>```, the videos are slowed four times with a framerate of 30 fps. The outputs are ***out_raw.mp4** and ***out_mask.mp4*** respectively. The Eulerian Motion Magnification is applied on these videos.
4. The **evm.py** takes raw and mask videos as input and outputs the amplified version of the raw video input.
## Algorithm
1. Read raw and mask videos.
2. Set the following parameters.
```python
alpha = 5               # amplification factor
w_l = 0.5               # lower_hertz (in Hz)
w_h = 10                # upper_hertz (in Hz)
sampling_rate = 30      # sampling rate
pyr_lvls = 4            # number of pyramid levels
```
3. Extract the following video metadata.
```python
n_frames = reader1.get_meta_data()['nframes']
width = reader1.get_meta_data()['size'][0]
height = reader1.get_meta_data()['size'][1]
fps = reader1.get_meta_data()['fps']
```
4. Discard blue and green channels from the raw video. This is done so that we can mark the amplification of the blood flow in vessels using red color.
5. Spatial decomposition using Gaussian pyramid.
6. Temporal processing on each spatial band by applying bandpass filter to extract frequency band of interest.
7. Amplify the output of previous step by a factor α.
8. Resize the video to original size.
9. Add amplified video to original video.
10. Postprocess the video.
```python
# Discarding amplified background
im1 = cv2.bitwise_or(vid,im_mas) 
# Taking the background from raw images
im2 = cv2.bitwise_or(im_raw,cv2.bitwise_not(im_mas))
# Adding the foreground and the background
im3 = cv2.bitwise_and(im1,im2)
```
11. Save output video to file.
## Flowchart
