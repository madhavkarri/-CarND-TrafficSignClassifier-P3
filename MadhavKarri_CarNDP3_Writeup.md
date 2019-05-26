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
[image6]: ./Writeup_IV/I4_CCNArch.png "I4_CCNArch"

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


### Design and Test a Model Architecture

### 1. Preprocess Image Data
Several data and image preprocessing steps/techniques were perfoemd on the original German Traffic Sign data set. This will ease the neural net clasifier and the associated optimizer to reach the global minimum (loss function).

* Step 1: Shuffle training data to increase randomness in the data
* Step 2: Convert color image to grayscale. It was observed conversion of color image to grayscale for this specific data set will lower/minimze the data size by reducing number of channels in the image without the loss of any features.
* Step 3: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm to modify low contrast and brigthness images. This increases numer of acceptable samples for a given class
* Step 4: Scale the processed image data to lie between 0 and 1 by dividing all the pixel values by 255. Will help numeircal stability and convergence during optimizationn process
* Step 5: Center data around zero by using mean and variance. The mean and variance are calcualted through all the images per pixel from the same position. As in step 4, this will also help numeircal stability and convergence during optimizationn process

#### 2. Model Architecture 
* The neural-net model selected for this classification was based on convolutional neural-net (CNN) developed by  Pierre Sermanet / Yann LeCun paper.
![][image6]
The primary ascpect of this architecture, "In traditional ConvNets, the output of the last stage is fed to a classifier. In the present work the outputs of all the stages are fed to the classifier. This allows the classifier to use, not just high-level features, which tend to be global, invariant, but with little precise details, but also pooled lowlevel features, which tend to be more local, less invariant,
and more accurately encode local motifs."

  
### Reflection

### Possible improvements to pipeline
- Automate the process either through Machine Learning/AI for the following set of tasks
  - To pick optimal co-ordinates for masking
  - For the prior detection of shadows, high brightness scenarios that masks yellow/white lane line markings
- Using mapping and localization explore possibility to detect and utilize priori information on lane coordinates and curvature
- Impelement an shadow detection algorithm such as [Link](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.95.3832&rep=rep1&type=pdf)

