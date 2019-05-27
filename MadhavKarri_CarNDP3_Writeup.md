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
[image2]: ./Writeup_IV/I2_DVE1.png "I2_DVE1"
[image3]: ./Writeup_IV/I3_SCTrain.png "I3_SCTrain"
[image4]: ./Writeup_IV/I3_SCValid.png "I3_SCValid"
[image5]: ./Writeup_IV/I3_SCTest.png "I3_SCTest"
[image6]: ./Writeup_IV/I4_CNNArch.png "I4_CNNArch"
[image7]: ./Writeup_IV/I5_CNNArch.png "I5_CNNArch"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

Project Python Code (Interactive python file: Traffic_Sign_Classifier_v7.ipynb)

Python Code/Implementation: [Link](./MadhavKarri-Project3-Files/Traffic_Sign_Classifier_v7.ipynb)


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


### Design and Test a Model Architecture

### 1. Preprocess Image Data
Several data and image preprocessing steps/techniques were perfoemd on the original German Traffic Sign data set. This will ease the neural net clasifier and the associated optimizer to reach the global minimum (loss function).

* Step 1: Shuffle training data to increase randomness in the data
* Step 2: Convert color image to grayscale. It was observed conversion of color image to grayscale for this specific data set will lower/minimze the data size by reducing number of channels in the image without the loss of any features.
* Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm to modify low contrast and brigthness images. This increases numer of acceptable samples for a given class
* Step 4: Scale the processed image data to lie between 0 and 1 by dividing all the pixel values by 255. Will help numeircal stability and convergence during optimizationn process
* Step 5: Center data around zero by using mean and variance. The mean and variance are calcualted through all the images per pixel from the same position. As in step 4, this will also help numeircal stability and convergence during optimizationn process. Step 5 was not implemented initially. This was a later addition to increase accuracy of the training process. Ideally, this step should have been implemented prior to the execution of neural-network

#### 2. Model Architecture 
* The neural-net model selected for this classification was based on convolutional neural-net (CNN) developed by  [Pierre Sermanet / Yann LeCun paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

![][image6]

The primary ascpect of this architecture, "In traditional ConvNets, the output of the last stage is fed to a classifier. In the present work the outputs of all the stages are fed to the classifier. This allows the classifier to use, not just high-level features, which tend to be global, invariant, but with little precise details, but also pooled lowlevel features, which tend to be more local, less invariant,
and more accurately encode local motifs."

This model was specifically selected, because of its application on German Traffic Sign Image Data Set.

Final model consisted of the following layer architecture:

![][image7]

#### 3. Model Training Parameters
The following training parameters were used during the training, validation, and testing process:
* Learning Rate = 0.001 (Reasoning: based on a few initial runs, 0.001 (default) seemed to be appropriate value for this study)
* Number of Steps (for optimizer): 4500-5000 (Reasoning: anything above 4500-5000 steps did not attain any further gains in performance)
* Batch Size: 125
* Dropout Probability: 0.5
* Optiization Technique: Adam Optimizer (Reasoning: succinctly adaptive learning rate algorithm)


#### 4. Solution Approach

Final model results were as follows:
* Validation set accuracy: 0.97
* Test set accuracy of: 0.957

Above numbers were arrived through an iterative process:
* Primary tuning parameter was the filter size on convolution layers. 
  * Initially ran the estimator using approximately 6 and 16 filters for convolution layers 1 and 2, this resulted in about 30-40% accuracy
  * Modified number of filters to 32 and 64 for convolution layers 1 and 2, this resulted in an increase of accuracy upto 80%
* Preprocess Image Data-Step 5 (averaging image data using mean and variance) resulted in a further increase of accuracy beyond 90%


### Test a Model on New Images

#### 1. Five German traffic signs found on the web:

[Label 8](https://www.bloomberg.com/opinion/articles/2019-01-27/autobahn-speed-limits-good-for-the-environment-bad-for-germany)

<img src="Writeup_IV/NGTS_1.png" width="100">

[Label 11](https://dc2ktown.files.wordpress.com/2013/08/blogpriroad.jpg)

<img src="Writeup_IV/NGTS_2.png" width="100">

[Label 12](https://angelikasgerman.co.uk/wp-content/uploads/2018/02/Priority.jpg)

<img src="Writeup_IV/NGTS_3.png" width="100">

[Label 13](https://angelikasgerman.co.uk/wp-content/uploads/2018/02/Vorfahrt-achten.jpg)

<img src="Writeup_IV/NGTS_4.png" width="100">

[Label 18](https://www.businessinsider.in/Minnesotas-governor-just-issued-a-sweeping-and-emotional-executive-order-about-vowels-on-street-signs/articleshow/46938741.cms)

<img src="Writeup_IV/NGTS_5.png" width="100">

Images with labels 11, 12, and 13 might potentially be difficult to classify, because of additional sign-boards sitting above the target traffic signs

#### 2. Model Predictions

* The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 
* This compares reasonably to the accuracy on the test set of 95.7%.
* Model missed predicting label 11. Potential reasons
  * Original image is at an angle
  * Original image when modified to 32X32 pixels, the image got further skewed
  
  [NGTS_IP]: ./Writeup_IV/NGTS_IP.png "NGTS_IP"
  
  ![][NGTS_IP]
  

#### 3. Top 5 Softmax Probabilities for Each Image
* Original image label input: [8 11 12 13 18]
* Class prediction: [8 23 12 13 18]
* Softmax Probabilities
* Image 1 (Groundtruth Value: 8)
  * Predicted Label and Probability
  * Label 8: 0.73
    * Label 1: 0.12
    * Label 4: 0.12
    * Label 5: 0.028
    * Label 0: 0.0047
* Image 2 (Groundtruth Value: 11)  
  * Predicted Label and Probability
  * Label 23: 0.72
    * Label 12: 0.1
    * Label 11: 0.08
    * Label 40: 0.03
    * Label 27: 0.02
* Image 3 (Groundtruth Value: 12) 
  * Predicted Label and Probability
  * Label 12: 1
    * Label 40: 0
    * Label 9: 0
    * Label 14: 0
    * Label 37: 0
* Image 4 (Groundtruth Value: 13)
  * Predicted Label and Probability
  * Label 13: 1
    * Label 12: 0
    * Label 25: 0
    * Label 28: 0
    * Label 38: 0
* Image 5 (Groundtruth Value: 18)
  * Predicted Label and Probability
  * Label 18: 1
    * Label 26: 0
    * Label 37: 0
    * Label 1: 0
    * Label 22: 0 

[NGTS_T5Pa]: ./Writeup_IV/NGTS_T5Pa.png "NGTS_T5Pa"
![][NGTS_T5Pa]

[NGTS_T5Pb]: ./Writeup_IV/NGTS_T5Pb.png "NGTS_T5Pb"
![][NGTS_T5Pb]


### Reflections and Further Improvements
There are many improvements that can be implemented to further increase the predcition accuracy
* A significant variation in sample count between different classes can be minimized. Data augumentation can be perfromed using exisiting data set and performing operations such as rotation, mirroring, skewing etc. Tools are available within tensorflow itself that accomplishes this task with minimal effort
* There is a potential chance to improve accuracy by implementing L2-Regularization

