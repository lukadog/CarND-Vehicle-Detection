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
[image4]: ./examples/search_area.png
[image5]: ./examples/combined_search_result.png
[image6]: ./examples/heat_map.png
[image7]: ./examples/threshold_heat_map.png
[image8]: ./examples/label.png
[image9]: ./examples/final_result.png
[image10]: ./examples/ezgif.com-video-to-gif.gif
[image11]: ./examples/ezgif.com-video-to-gif-2.png


Rubric Points
---

# Histogram of Oriented Gradients (HOG)


## Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

Step 1. Load all of the car and non-car image paths from the provided dataset. The figure below shows 8 sample images for each class.

![alt text][image1]

Step 2. Extract HOG features from an training image using function `extract_hog_features`. The figures below show a car image with its associated hog features as well as a non-car image with its associated hog features.

![alt text][image2]
![alt text][image3]

Step 3. Training the SVM classifier with different parameter combinations. Below table captures different combinations that I have tried and their accuracy. It shows `|YUV | 8 | 8 | 2 | 0.9982|` give the best result.

```
for orient in range(8, 10):
		for pix_per_cell in range(8, 10):
			for cell_per_block in range(2, 4):
				for colorspace in ['RGB', 'YUV']:
						X_train, X_test, y_train, y_test = construct_data_set(car_images, noncar_images, colorspace, orient, pix_per_cell, cell_per_block, hog_channel='ALL')
						svc = train_classifier(X_train, y_train)
						print(colorspace + ' | ' + str(orient) + ' | ' + str(pix_per_cell) + ' | ' + str(cell_per_block) + ' | ' + str(round(svc.score(X_test, y_test), 4)))

```


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

## Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Step 1. Construct a training dataset using function `construct_data_set`. It returns a shuffled training set and testing set.

Step 2. Train the linear SVM using `train_classifier` function. The trained model is saved as `svc`

Note: no color information is used as I don't believe it's correlated to car features.

# Sliding Window Search

## Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in `detect_cars` function. 

Step 1. The HOG features are extracted for image and fed to the classifier.

Step 2. Classifier makes prediction on the HOG features for each window and returns a list of rectangle coordinates that are predicted to have a car.

Step 3. Several different scales (1x, 1.5x, 2x and 3.5x) with various overlaps are applied. 

Below figure shows the search area of all windows:

![alt text][image4]

Step 4. The figure below shows the final result returned by `detect_cars` with different rectangle sizes. 

![alt text][image5]


## Show some examples of test images to demonstrate how your pipeline is working. How did you optimize the performance of your classifier?

First step is searching and predicting. To optimize the performance, we only search the area of the pavement. The results are shown in above figures.

After we have the rectangles, we generate heatmaps by adding 1 to the blocks inside the rectangles. To remove false positive (or false negative), we store the rectangles detected from the previous frame in `prev_rectangles` and let it contribute to the next frame. 

![alt text][image6]

Then we do thresholding based on the heatmap value.

![alt text][image7]

Then we label the data.

![alt text][image8]

Finally, we plot the bounding box based on the labeled data.

![alt text][image9]

# Video Implementation

## Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

![alt text][image10]
![alt text][image11]

## Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The entire pipeline is in `detect_cars` function. To reduce false positve, I have done below things:

1. Apply threshold to heatmap
2. Introduce `prev_rectangles` to current heatmap
3. Draw bounding box based on the labeled data as apposed to all rectangles.

## Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

1. Before use `prev_rectangles`, I have a lot of false positives
2. The training data has more than 99% accuracy for the test data set. However, it doesn't achieve that high accuracy for the video data. Maybe due to brightness or resolution differences.
3. The performance is still very slow, even given that I only searched the bottom area.
4. The searching block is not adaptive, so the bounding box is not very accurate.
5. I probabaly can try the Yolo network to compare the performance.



