# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
  

The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/visualize_training_data.png
[image2]: ./examples/car_hog.png
[image3]: ./examples/no_car_hog.png
[image4]: ./examples/visualize_training_data.png
[image5]: ./examples/visualize_training_data.png

Rubric Points
---

## Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

Step 1. Load all of the car and non-car image paths from the provided dataset. The figure below shows 8 sample images for each class.

![alt text][image1]

Step 2. Extract HOG features from an training image using function `get_hog_features`. The figures below show a car image with its associated hog features as well as a non-car image with its associated hog features.

![alt text][image2]
![alt text][image3]

Step 3. Training the SVM classifier with different parameter combinations. Below table captures different combinations that I have tried and their accuracy. It shows `|YUV | 8 | 8 | 2 | 0.9982|` give the best result.


| Color Space | Orient | Pixels Per Cell | Cells Per Block | Accuracy |
| :---------: | :----: | :-------------: | :-------------: | :-------:|
|YUV | 8 | 8 | 2 | 0.9982|
|RGB | 8 | 8 | 2 | 0.9909|
|RGB | 8 | 8 | 3 | 0.9914|
|YUV | 8 | 8 | 3 | 0.9977|
|RGB | 8 | 9 | 2 | 0.9896|
|YUV | 8 | 9 | 2 | 0.9973|
|RGB | 8 | 9 | 3 | 0.9837|
|YUV | 8 | 9 | 3 | 0.9955|
|RGB | 9 | 8 | 2 | 0.9928|
|YUV | 9 | 8 | 2 | 0.9964|
|RGB | 9 | 8 | 3 | 0.986 |
|YUV | 9 | 8 | 3 | 0.9968|
|RGB | 9 | 9 | 2 | 0.9878|
|YUV | 9 | 9 | 2 | 0.9964|
|RGB | 9 | 9 | 3 | 0.9869|
|YUV | 9 | 9 | 3 | 0.9955|



