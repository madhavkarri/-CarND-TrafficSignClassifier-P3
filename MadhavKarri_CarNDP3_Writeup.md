# **Self-Driving Car Engineer Nanodegree**
# **Project3: Build a Traffic Sign Recognition Classifier**

## MK

Overview
---
In this project, use deep neural networks and convolutional neural networks to classify traffic signs. Train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). Use the trained model to predict and classify German traffic sign images from the web.

The Project
---
The goals/steps for this project:
* Load the German Traffic Sign data set
* Explore, summarize, and visualize the data set
* Design, train, and test a neural network model architecture
* Use neural network model to make predictions on new German Traffic Sign images from the web
* Analyze the softmax probabilities of the new images
* Summarize and reflect on your work in a written report

---

[//]: # (Image References)

[image1]: ./Writeup_IV/I1_DataSummary.png "I1_DataSummary"
[image2]: ./examples/I2_DVE1.png "I2_DVE1"
[image3]: ./examples/I3_SCTrain.png "I3_SCTrain"
[image4]: ./examples/I3_SCValid.png "I3_SCValid"
[image5]: ./examples/I3_SCTest.png "I3_SCTest"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

Project Python Code (Interactive python file: Traffic_Sign_Classifier_v6.ipynb)

Python Code/Implementation: [Link](./MadhavKarri-Project3-Files/Traffic_Sign_Classifier_v6.ipynb)


### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Used numpy and pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

![][image1]

#### 2. Include an exploratory visualization of the dataset.

An exploratory visualization of the data set was performed on the entire data set. Each traffic sign was dsiplayed and listed with the following set fo features:

* Traffic Sign id:
* Traffic Sign Label: 
* Traffic Sign 0 Training Sample Count : 
* Traffic Sign 0 Training Sample Distribution : 
* Display 5 sample image for each class/label from training set data

![][image2]

Below plots show sample count for each of the classes from the training, validation, and test sets

![][image3]
![][image4]
![][image5]






**Finding Lane Lines on Test Images/Frames**

Pipeline for test images (Interactive python file: Pipeline_Test_Images.ipynb)

Python Code/Implementation: [Link](./MadhavKarri-Project2-Files/Pipeline_Test_Images.ipynb)

* Load all neceaary python imports (cell: 1)
* Implement Camera Calibration (cell: 2)
  - Camera calibration was performed using provided chess board images
  - Recommended number of images for a good calibration is about 20
  - However, few of the provided images, images 1,4 and 5, did not fit/match 9X6 pattern
  - Sequence of steps followed for camera calibration:
    - Convert image to grayscale
    - Find corners of chessboard using cv2 function "findchessboardcorners"
    - Example: find chessboard corners-calibration12.jpg
    ![FindChessboardCorners_TI12](./Writeup_IV/FindChessboardCorners_TI12.jpg)
    - Create object points in 3d plane and image points. Image points are corners of the chessboard determined in the previous step
    - Calibrate camera using object points and image points using cv2 function "calibrateCamera"
    - "calibrateCamera" function outputs camera calibration matrix and distortion coefficients
    - The original image can be undistorted using camera calibration matrix and distortion coefficients
    - Final step: apply perspective transform matrix on the undistorted image to get birds eyeview
    - Example: perspective transform-calibration12.jpg
    ![UndistortedPerspectiveTransform_TI12](./Writeup_IV/UndistortedPerspectiveTransform_TI12.jpg)
    - Calibration error can be computed using cv2 function "projectionPoints", and taking L2-Norm between the image poinst and output from cv2 function "projectionPoints"
    - The error initially is quite low on the order of magnitude 10^-2, but increases proprotional to the number of images and consequently will be asymptotic. 
    - The calibration error for the given test images inside the "camera_cal" folder is about approximately 2
    - The resulting camera matrix and distortion coefficients are as follows
    ![CameraCal_CMDC](./Writeup_IV/CameraCal_CMDC.jpg)
* Apply camera matrix and distortion coefficients to undistort each frame (function: frame_ud)
* Apply color selection binary by thresholding S-channel of HLS color space (function: hls_select)
  - HLS Color Space with S-Channel was determined to be the appropriate binary for detecting yellow lane lines under varying brightness or shades
- Apply gradient, gradient-magnitude, and gradient-direction thresholds using sobel operator
  - Gradient Threshold (function: grad_thresh)
  - Gradient Magnitude Threshold (function: mag_thresh)
  - Gradient Direction Threshold (function: dir_thresh)
  - Gradient-x and y will result in highlighting line-edges features closer to vertical and horizontal directions respectively
* Apply masking and perspective transform (function: frame_mptw)
  - Masking was applied by picking 4-corner co-ordinates of a trapezoid close enough to the target area on the lane
  - The perspective transform matrix was determined by picking the right set of source and destination points and cv2 function "getPerspectiveTransform".
    - For source points the 4-corner co-ordinates of the trapezoidal mask were picked
    - For detsination points the four corner co-ordinates of the image boundary with margin were selected
* Find lane starting co-ordinates on X-axis by taking histogram of detected lane line pixels on bottom-half of the warped binary output (function: wi_hp)
  - A histogram of the lane line pixels on the bottom half of the frame/image in the target area will result in two peaks. These two peaks, corresponding to left and right lane lines, can be used as a good starting guess for the sliding window algorithm.
* Post establishing the starting co-ordinates, of the detected lane line pixels, on x-axis, implement the sliding window algorithm to detect lane line pixels through all of the image/frame (function: find_lane_pixels)
* Post detetction of left and right line pixels through the complete image, a polynomial fit is established for each of the left and right lines (function: fit_polynomial)
* Implement band/margin search around line position from previous frame, equivalent to using a customized region of interest for each frame of video. In the unlikely case, lost track of the lines, revert back to sliding windows search (function: search_around_poly)
* Measure radius of curvature for the lane and offset of center of vehicle from center of lane (function: measure_curvature_real)
* Final Step: Map/Draw detected lane on to the original undistorted frame/image (function: map_lane_udi)

The above set of steps were implemented on the following set of test images
* Test1.jpg
* Test2.jpg
* Test3.jpg
* Test4.jpg
* Test5.jpg
* Test6.jpg
* Straight_lines1.jpg
* Straight_lines2.jpg

Results from Pipeline for test images on Test3.jpg are shown below:
* Applying camera matrix and distortion coefficients to undistort each frame
![Test3_Undistorted](./Writeup_IV/Test3_Undistorted.jpg)
* Apply color selection binary by thresholding S-channel of HLS color space
![Test3_HLSBinary](./Writeup_IV/Test3_HLSBinary.jpg)
* Apply gradient, gradient-magnitude, and gradient-direction thresholds using sobel operator
![Test3_GMDBinary](./Writeup_IV/Test3_GMDBinary.jpg)
* Apply masking and perspective transform
![Test3_MPTBinary](./Writeup_IV/Test3_MPTBinary.jpg)
* Find lane starting co-ordinates on X-axis by taking histogram of detected lane line pixels on bottom-half of the warped binary output
![Test3_HPeaks](./Writeup_IV/Test3_HPeaks.jpg)
* Implement sliding window algorithm to detect lane line pixels through all of the image/frame 
* Perform a polynomial fit on the detetcted lane line pixels for each of the left and right lines
![Test3_SWPF](./Writeup_IV/Test3_SWPF.jpg)
* Implement band/margin search around previously detected left and right lines from previous frame
![Test3_SAPC](./Writeup_IV/Test3_SAPC.jpg)
* Measure radius of curvature for the lane and offset of center of vehicle from center of lane
![Test3_CurvRad](./Writeup_IV/Test3_CurvRad.jpg)
* Final Step: Map/Draw detected lane on to the original undistorted frame/image (function: map_lane_udi)
![Test3_LDetect](./Writeup_IV/Test3_LDetect.jpg)

**Finding Lane Lines in a Video (Project_Video)**

- The pipeline/steps implemented for finding lane lines in a video are similar to finding lane lines on test images
  - Implemented a python code to extract raw frames/static images from a video file
  - Implemented a pipeline consisting primarily of 1 main function and all the functions previously listed in the descripition of finding lane lines on test images.
  - Main Function/Wrapper: A while loop that calls all the necessary functions repeatedly on each of the image frames extracted from the original video.
  - Implemented a python code to stich final output frames from the preceeding steps and convert it into a video
  
Additional Comments
- A line class has been implemented which keeps tracks of all the line charachteristics (fit coefficients, radius of curvature, vehicle center offset, etc.) between frames
- An attempt has been made to avergae the polynomial fit coefficients  between n frames. The averaging has been performed such that for the nth frame, frames from n-10 to n+10 were used to average the polynomial coefficients.
- Smoothing: the objective of the averaging technique was to minimize the jumping of lane lines from frame to frame
- Surprisingly, the video output without averaging performed better than the one with averaging. Likely cause might be the number of frames being considered for averaging might be too high
- The below final output video is one without averaging

**Project Video Output**

 - Project Video
   - Python Code/Implementation: [Link](./Writeup_IV/Pipeline_Project_Video.ipynb)
   - Video Output File (YouTube): [Link](https://youtu.be/wEArUuQCSfM)

---

### Optional Challange

- The pipeline/steps implemented for finding lane lines for the challenge video are similar to finding lane lines on the project video
  - Implemented a python code to extract raw frames/static images from a video file
  - Implemented a pipeline consisting primarily of 1 main function and all the functions previously listed in the descripition of finding lane lines on test images.
  - Main Function/Wrapper: A while loop that calls all the necessary functions repeatedly on each of the image frames extracted from the original video.
  - Implemented a python code to stich final output frames from the preceeding steps and convert it into a video
  
Additional Comments
- First major challenge in this video was to detect/identify and remove the misealding/spurious lane markings as shown below
![ChallengeVideo_MSLM](./Writeup_IV/ChallengeVideo_MSLM.jpg)
- The misealding/spurious lane marking are detected using HSV colorspace. By adjusting the thresholds, the HSV binary was able to detect all the lanes and its features except for the yelllow and white lane markings. The result is shown below
![UD_HSV17](./Writeup_IV/UD_HSV17.jpg)
- The solution was to subtract the HSV binary from the undistorted image to remove the misealding/spurious lane markings
- The subtraction was performed post applying - Gradient, Gradient Magnitude, and Gradient Direction Thresholds
- Second major challenge in this video was to detect/identify and remove shadow that was making the yellow and white lane line markings as shown below
![cv_frame130](./Writeup_IV/cv_frame130.jpg)
- The shadow region under the bridge was addressed through a combination of HSV and RGB colorspaces and CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm

**Challenge Video Output**

 - Challenge Video
   - Python Code/Implementation: [Link](./Writeup_IV/Pipeline_Challenge_Video.ipynb)
   - Video Output File (YouTube): [Link](https://youtu.be/jzV-YqbOaFw)
   - Video Output File: [Link](./Writeup_IV/ChallengeVideo_fo.mp4)
  
### Reflection

### Potential shortcomings with the current pipeline
- A lot of manual processing is involved with the exisiting pipeline.
  - Manual picking of masking co-ordinates
  - Fine tuning parameters for each of the thersholds based on the change in frame scene: shadows, brightness, etc.

### Possible improvements to pipeline
- Automate the process either through Machine Learning/AI for the following set of tasks
  - To pick optimal co-ordinates for masking
  - For the prior detection of shadows, high brightness scenarios that masks yellow/white lane line markings
- Using mapping and localization explore possibility to detect and utilize priori information on lane coordinates and curvature
- Impelement an shadow detection algorithm such as [Link](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.95.3832&rep=rep1&type=pdf)

