# COMPUTER VISION PROJECT
## RETINAL BLOOD FLOW MAGNIFIATION USING EULERIAN VIDEO MAGNIFICATION
The project presents the application of Eulerian Video Magnification on retinal video angiographic images.

Steps:
1. Preprocess fluorescein angiogram (FA) images. Preprocessing involves proper alignment of images and image segmentation to separate blood vessels from background. Preprocessimg steps and data are not shown in this project. Separately mask images are also created for respective FA images.
2. The preprocessed images are then converted to video. In our example we had 54 FA images. The output video has a framerate of 3 frames per second and a duration of 18 seconds. Similarly a mask video with same properties is also created. These are ***op_raw.mp4*** and ***op_masks.mp4*** respectively.
3. 
