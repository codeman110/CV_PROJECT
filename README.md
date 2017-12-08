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
# Parameters
alpha = 5               # amplification factor
w_l = 0.5               # lower_hertz (in Hz)
w_h = 10                # upper_hertz (in Hz)
sampling_rate = 30      # sampling rate
pyr_lvls = 4            # nmbr of pyramid levels
```
