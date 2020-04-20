# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/unique_signs_train.png "Training Set Unique Signs"
[image2]: ./examples/unique_signs_valid.png "Validation Set Unique Signs"
[image3]: ./examples/unique_signs_test.png "Test Set Unique Signs"
[image4]: ./examples/trafficsigns_random.png "exploratory visualization"
[image5]: ./examples/train_set_accuracy.png "Accuracy training set"
[image6]: ./examples/valid_set_accuracy.png "Accuracy validation set"
[image7]: ./examples/chosen_images_from_web.png "Chosen Traffic Sign Images"


---
### Writeup / README

Here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. There are bar charts showing how many signs there are per label-id in training-, validation and test-set. 

![alt text][image1]
![alt text][image2]
![alt text][image3]

Here you can see randomly chosen signs for visualization from the dataset:

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Preprocessing of the image data. 

As a first step, I decided to convert the images to grayscale because the image-grayscaling will help in sharpening the image by identifying light and dark areas, so the model can better learn from the geometry present in the image.

As a last step, I normalized the image data because it ensures that the pixel parameter has an equal data distribution and so it makes the convergence faster while training the network. When there is no normalization applied, the ranges of distributions of feature values would be different for each feature and so the learning rate would cause corrections in each dimension.

#### 2. Final model architecture (LeNet architecture):

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image  						| 
| Convolution       	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Activation			| RELU											|
| Pooling	        	| 2x2 stride, valid padding  outputs 14x14x6 	|
| Convolution   	    | 1x1 stride, valid padding, outputs 10x10x16	|
| Activation			| RELU											|
| Pooling	        	| 2x2 stride, valid padding  outputs 5x5x16 	|
| Flatten   			| Output = 400									|
| Fully Connected  	    | Output = 120									|
| Activation			| RELU											|
| Fully Connected  	    | Output = 84									|
| Activation			| RELU											|
| Fully Connected  	    | Output = 43									|



#### 3. Training of the model. (optimizer, batch size, number of epochs, learning rate)

To train the model, I used an AdamOptimizer and chose a batch size of 156, 28 epochs and a learning rate of 0.002. I got better results with this higher learning rate, so the training started with a validation accuracy of 0.800 and steadily increased overall.

#### 4. Approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.940
* test set accuracy of 0.924

![alt text][image5]
![alt text][image6]

You find the calculation of accuracies in the Code cell with the Heading *Train, Validate and Test the Model* with the function `evaluate(X_data, y_data)`

At first I tried out less epochs and a lower learning rate. But the best results gave a higher epoch number and a higher learning rate.

I sticked with the LeNet-architecture because it is a successful Gradient-Based Learning technique and it has a good classification of high-dimensional patterns such as handwritten letters or like in this example forms and attributes of traffic sign. It also provided best results in accuracy for million of hand written checks and it provided also good results for recognizing traffic signs.

The training, validation and test set accuracy are all over 92 %, where the test set has the lowest accuracy and the training set the highest accuracy with 99.9 %. This good results prove that the model is working well.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image7]

The first two images might be difficult to classify because they are not photographed plainly. The first was photographed from the left and the second from the bottom. They also had both something overwritten in full size picture.

#### 2. Model's predictions on these new traffic signs and comparision of the results to predicting on the test set. .

Here are the results of the prediction: 

| Image			                            | Prediction	       			| 
|:-----------------------------------------:|:-----------------------------:| 
| Speed limit (50km/h)						| Speed limit (20km/h)			| 
| Road narrows on the right					| Slippery road 				|
| Right-of-way at the next intersection		| correct						|
| Yield	      								| correct		 				|
| Priority road								| correct    					|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This accuracy is much lower than the accuracy of the test set probably because the first two pictures I chose where taken at a very difficult camera angle.

#### 3. Looking at the softmax probabilities for each prediction (top 5 softmax probabilities for each image along with the sign type of each probability).

The code for making predictions on my final model is located in the second last cell of the Ipython notebook.

For the first image, the model is very sure that this is a Speed limit of 20 km/h (probability of 1.0), but the image does contains a Speed limit of 50 km/h . The top five soft max probabilities were

Image1:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (20km/h)							| 
| 3.54225e-15 			| Speed limit (30km/h)							|
| 3.8059e-19			| Speed limit (70km/h)							|
| 4.7525e-21  			| End of speed limit (80km/h)	 				|
| 1.27024e-22		    | Dangerous curve to the right					|


For the second image, the model is also sure that this is the traffic sign Slippery road (probability of 0.926), but the image does contains the traffic sign Road narrows on the right . The top five soft max probabilities were

Image2:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.925891  			| Slippery road									| 
| 0.074106 				| Keep right									|
| 2.49624e-06 			| Road work										|
| 4.96584e-07 			| Dangerous curve to the right	 				|
| 2.00072e-09		    | No passing for vehicles over 3.5 metric ton	|

For the third, fourth and fifth image, the model is very sure (each probability of 1.0) that on the images there are the signs Right-of-way at the next intersection, Yield and Priority road, which all are correctly predicted.

Here are the top five soft max probabilities for these images:

Image3:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0  					| Right-of-way at the next intersection			| 
| 1.12986e-07			| Beware of ice/snow							|
| 7.27848e-18 			| Road work										|
| 1.16547e-18   		| Double curve	 								|
| 1.08911e-19 		    | Slippery road									|

Image4:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0  					| Yield											| 
| 3.68666e-22			| Priority road									|
| 3.79154e-23  			| Ahead only									|
| 9.6549e-26 			| Keep left	 									|
| 3.96744e-27		    | Go straight or right							|

Image5:
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0  					| Priority road									| 
| 9.97825e-14			| Roundabout mandatory							|
| 2.1656e-16   			| Ahead only									|
| 3.8347e-18 			| No passing 									|
| 8.92033e-19		    | Beware of ice/snow							|



