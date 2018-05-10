from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import numpy as np
import pickle
import cv2
import glob
import time


def load_training_data():
	# load data from training data set
	car_images = glob.glob('training_dataset/car/*.png')
	noncar_images = glob.glob('training_dataset/nocar/*.png')
	# print(len(car_images), len(noncar_images))
	return car_images, noncar_images


def visualize_training_data(car_images, noncar_images):
	# create a canvas size of 4 X 4 to display sample data
	fig, axs = plt.subplots(4, 4, figsize=(8, 8))
	fig.subplots_adjust(hspace = .3, wspace=.005)
	axs = axs.ravel()
	# randomly plot training data set
	for i in np.arange(8):
		img = cv2.imread(car_images[np.random.randint(0, len(car_images))])
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		axs[i].axis('off')
		axs[i].set_title('car', fontsize=10)
		axs[i].imshow(img)
	for i in np.arange(8,16):
		img = cv2.imread(noncar_images[np.random.randint(0,len(noncar_images))])
		img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		axs[i].axis('off')
		axs[i].set_title('no_car', fontsize=10)
		axs[i].imshow(img)
	plt.show()


def get_hog_features(gray_img, orient, pix_per_cell=8, cell_per_block=2, vis=False, feature_vec=True):
	# given a gray scale image, return the hog features
	# if vis==True, return both hog features and hog_image for visualization
	if vis == True:
		features, hog_image = hog(gray_img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
		cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis, feature_vector=feature_vec)
		return features, hog_image
	# if vis==False, only return hog features
	else:
		features = hog(gray_img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
		cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, visualise=vis, feature_vector=feature_vec)     
		return features


def visualize_hog_image(img):
	# visualize the original image and its hog features
	gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY )
	features, hog_image = get_hog_features(gray_img, orient=9, pix_per_cell=8, cell_per_block=8, vis=True, feature_vec=True)
	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
	f.subplots_adjust(hspace = .3, wspace=.2)
	ax1.imshow(img)
	ax1.set_title('no_car image', fontsize=10)
	ax1.axis('off')
	ax2.imshow(hog_image, cmap='gray')
	ax2.set_title('hog features', fontsize=10)
	ax2.axis('off')
	plt.show()


def extract_hog_features(imgs, cspace='RGB', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0):
	# define a features list to append feature vectors 
	features = []
	# iterate through the list of imgs
	for file_path in imgs:
		img = mpimg.imread(file_path)
		# Convert image to new color space if specified
		if cspace != 'RGB':
			if cspace == 'HSV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
			elif cspace == 'LUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
			elif cspace == 'HLS':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
			elif cspace == 'YUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
			elif cspace == 'YCrCb':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
		else: feature_image = np.copy(img)      
		# call get_hog_features() with vis=False, feature_vec=True
		if hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features = np.append(hog_features, get_hog_features(feature_image[:,:,channel], 
					orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
				# hog_features.append(get_hog_features(feature_image[:,:,channel], 
				# 	orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)      
		else:
			hog_features = np.append(hog_features, get_hog_features(feature_image[:,:,channel], 
				orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True))
			# hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
			# 	pix_per_cell, cell_per_block, vis=False, feature_vec=True)
		# append the feature vector to the features list
		features.append(hog_features)
		# Scale the feature vector
		# X_scaler = StandardScaler().fit(features)
		# scaled_X = X_scaler.transform(features)
	# return list of feature vectors
	return features


def construct_data_set(car_images, noncar_images, colorspace='YUV', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel = 'ALL'):
	# create features for the car images
	car_features = extract_hog_features(car_images, cspace=colorspace, orient=orient, 
		pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
	# create features for the no car images
	notcar_features = extract_hog_features(noncar_images, cspace=colorspace, orient=orient, 
		pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel)
	# stack features vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64) 
	# create labels vector 
	y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
	# split up data into training and test sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=np.random.randint(0, 100))
	print('length of training feature vector: ' + str(len(X_train)))
	print('length of training label vector: ' + str(len(y_train)))
	print('length of testing feature vector: ' + str(len(X_test)))
	print('length of testing label vector: ' + str(len(y_test)))

	return X_train, X_test, y_train, y_test


def train_classifier(X_train, y_train):
	# define a linear SVC classifier 
	svc = LinearSVC()
	svc.fit(X_train, y_train)
	return svc


def detect_cars(img, ystart, ystop, scale, cspace, hog_channel, svc, orient, pix_per_cell, cell_per_block):
	# define an array of rectangles where cars were detected
	rectangles = []
	img = img.astype(np.float32)/255
	img_to_search = img[ystart:ystop,:,:]
	# convert image to new color space if specified
	if cspace != 'RGB':
		if cspace == 'HSV':
			feature_image = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2HSV)
		elif cspace == 'LUV':
			feature_image = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2LUV)
		elif cspace == 'HLS':
			feature_image = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2HLS)
		elif cspace == 'YUV':
			feature_image = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2YUV)
		elif cspace == 'YCrCb':
			feature_image = cv2.cvtColor(img_to_search, cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(img_to_search)  
	# rescale image if other than 1.0 scale
	if scale != 1:
		imshape = feature_image.shape
		feature_image = cv2.resize(feature_image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		# select colorspace channel for extracting HOG features 
	if hog_channel == 'ALL':
		ch1 = feature_image[:,:,0]
		ch2 = feature_image[:,:,1]
		ch3 = feature_image[:,:,2]
	else: 
		ch1 = feature_image[:,:,hog_channel]

	# define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell)+1  
	nyblocks = (ch1.shape[0] // pix_per_cell)+1 
	nfeat_per_block = orient*cell_per_block**2

	# define window size
	window = 64
	nblocks_per_window = (window // pix_per_cell)-1 
	cells_per_step = 2  
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step

	# compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)   
	if hog_channel == 'ALL':
		hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
		hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

	for xb in range(nxsteps):
		for yb in range(nysteps):
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# extract HOG feature for this patch
			hog_feature = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
			if hog_channel == 'ALL':
				hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
				hog_features = np.hstack((hog_feature, hog_feat2, hog_feat3))
			else:
				hog_features = hog_feature

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell
			# make the prediction
			test_prediction = svc.predict([hog_features])
			# if the window is predicted to be have car, append the coordinates of rectangle to rectangles
			if test_prediction[0] == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				rectangles.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
	# return coordinates of all detected rectangles
	return rectangles



def draw_boxes(img, bboxes, color=(0, 255, 0), thick=3):
	# make a copy of the image
	box_image = np.copy(img)
	# iterate through the bounding boxes
	for bbox in bboxes:
		# draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# return the image with bounding box 
	return box_image


def add_heat(heatmap, bbox_list):
	for box in bbox_list:
		# ddd += 1 for all pixels inside each bbox
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
	return heatmap

def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    rects = []
    for car_number in range(1, labels[1]+1):
        # find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        rects.append(bbox)
        # draw the bounding box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 3)
    return img, rects

def process_pipeline(img):
	# define pipeline parameters
	cspace = 'YUV'
	orient = 9
	pix_per_cell = 8
	cell_per_block = 2
	hog_channel = 'ALL'
	rectangles = []

	# Small sliding window range
	ystart = 400
	ystop = 464
	scale = 1.0
	rectangles.append(detect_cars(img=img, ystart=ystart, ystop=ystop, scale=scale, cspace=cspace, hog_channel=hog_channel, svc=svc, 
	                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block))
	# Small sliding window range
	ystart = 416
	ystop = 480
	scale = 1.0
	rectangles.append(detect_cars(img=img, ystart=ystart, ystop=ystop, scale=scale, cspace=cspace, hog_channel=hog_channel, svc=svc, 
	                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block))
	# Medium sliding window range
	ystart = 400
	ystop = 496
	scale = 1.5
	rectangles.append(detect_cars(img=img, ystart=ystart, ystop=ystop, scale=scale, cspace=cspace, hog_channel=hog_channel, svc=svc, 
	                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block))
	# Medium sliding window range
	ystart = 432
	ystop = 528
	scale = 1.5
	rectangles.append(detect_cars(img=img, ystart=ystart, ystop=ystop, scale=scale, cspace=cspace, hog_channel=hog_channel, svc=svc, 
	                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block))
	# Big sliding window range
	ystart = 400
	ystop = 528
	scale = 2.0
	rectangles.append(detect_cars(img=img, ystart=ystart, ystop=ystop, scale=scale, cspace=cspace, hog_channel=hog_channel, svc=svc, 
	                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block))
	# Big sliding window range
	ystart = 432
	ystop = 560
	scale = 2.0
	rectangles.append(detect_cars(img=img, ystart=ystart, ystop=ystop, scale=scale, cspace=cspace, hog_channel=hog_channel, svc=svc, 
	                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block))
	# Bigger sliding window range
	ystart = 400
	ystop = 596
	scale = 3.5
	rectangles.append(detect_cars(img=img, ystart=ystart, ystop=ystop, scale=scale, cspace=cspace, hog_channel=hog_channel, svc=svc, 
	                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block))
	# Bigger sliding window range
	ystart = 464
	ystop = 660
	scale = 3.5
	rectangles.append(detect_cars(img=img, ystart=ystart, ystop=ystop, scale=scale, cspace=cspace, hog_channel=hog_channel, svc=svc, 
	                       orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block))

	# reformat rectangles into a list
	rectangles = [item for sublist in rectangles for item in sublist] 

	# define prev_rectangles to store previous detected rectangles
	global prev_rectangles

	# add new detections to the prev_rectangles
	if len(rectangles) > 0:
		prev_rectangles.append(rectangles)
		if len(prev_rectangles) > 20:
			# remove the oldest 10 rectangles in the list 
			prev_rectangles = prev_rectangles[len(prev_rectangles) - 20:]

	heatmap_img = np.zeros_like(img[:,:,0])

	for rect in prev_rectangles:
		heatmap_img = add_heat(heatmap_img, rect)
		heatmap_img = apply_threshold(heatmap_img, 1)
	# label the heatmap
	labels = label(heatmap_img)
	# draw bounding box on original image
	draw_img, rect = draw_labeled_bboxes(np.copy(img), labels)
	
	return draw_img



if __name__ == '__main__':
	# load raw data set
	car_images, noncar_images = load_training_data()

	prev_rectangles=[]

	########################################
	# uncomment this part for visualization 
	########################################
	# visualize_training_data(car_images, noncar_images)
	# car_img = mpimg.imread(car_images[2])
	# visualize_hog_image(car_img)
	# nocar_img = mpimg.imread(noncar_images[5])
	# visualize_hog_image(nocar_img)

	# prepare training and testing data

	X_train, X_test, y_train, y_test = construct_data_set(car_images, noncar_images, colorspace='YUV', orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL')
	svc = train_classifier(X_train, y_train)

	cap = cv2.VideoCapture('project_video.mp4')

	while(cap.isOpened()):
		ret, frame = cap.read()
		if ret == True:
			img_out = process_pipeline(frame)
			cv2.imshow('image', img_out)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break






