#**Traffic Sign Recognition** 

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

[image1]: ./random-web-images/label_1.jpg "Traffic Sign 1"
[image2]: ./random-web-images/label_17.jpg "Traffic Sign 2"
[image3]: ./random-web-images/label_18.jpg "Traffic Sign 3"
[image4]: ./random-web-images/label_25.jpg "Traffic Sign 4"
[image5]: ./random-web-images/label_28.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.

I used the pandas library to print the shape of the image dataset:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.

I have performed the exploratory visualisation by displaying the first 20 images of each label found in the dataset.

I have used two techniques in order to do so:
- A fast method: Displays the first 20 images of one label under the same subplot `fast_draw_all_traffic_signs_in_different_images()`
- A slow method: Displays the first 20 images of all labels one after the other in one unique subplot `slow_draw_all_traffic_signs_in_one_image()`

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

The pre-processing includes:
- Transform all the images to grey scale with `x = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in x]`
 - This is convinient for our scenario as the colour of the images do not add key information of the detection of the signs
 - Reduces the amount of information by 3 that our network has to process so that we have faster processing and a more stable network
- Reshape all the images to conform with the expected data type for our network with `np.reshape(x, (-1, 32, 32, 1))`

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The loading of the training, validation and test sets are done in the first cell.
- The training set contains all the original 34799 samples and is useded fully shuffled each time before starting a training session
- The validation (size of 4410) and test sets (12630) are used only for evaluation of the model.

This setup produces an accuracy of 0.925 so that no further splitting of the training set was necessary

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the fifth of the ipython notebook.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 3x3     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, VALID padding, outputs 14x14x6	|
| Convolution 3x3	    | 1x1 stride, VALID padding, outputs 10x10x16	|
| RELU					| etc.        									|
| Max pooling			| 2x2 stride, VALID padding, outputs 5x5x16		|
| Flatten				| Outputs 400									|
| Fully connected		| Outputs 120									|
| RELU					|												|
| Fully connected		| Outputs 84									|
| RELU					|												|
| Fully connected		| Outputs 43


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the sixth cell of the ipython notebook.

To train the model, I used an AdamOptimizer along with a softmax with a one_hot configuration.

In order to produce fast an accurate prediction, the training set is randomised and splitted into batches of 128 images, and the full training runs 10 times (epochs).

The learning rate for the optimiser is set to 0.001

Also, the weight and bias are randomised using a mean of 0 and a standard deviation of 0.1

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the seventh cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.983
* validation set accuracy of 0.890
* test set accuracy of 0.879

A well known architecture was chosen:
* I choose the LaNet well known architecture in order to train the system for traffic signs
* After doing some research for valid architectures for traffic sign detection, the state-of-the-art indicates that LeNet is a good candidate for generalising the features of the traffic signs in a variarety of scenarios (as exposed here http://publications.lib.chalmers.se/records/fulltext/238914/238914.pdf)
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 * The training never gets as input the validation or test sets. After training the system with only the training set, the accuracy calculated against the validation and test set is above 89%

Iterations on the hyper-parameters:
- Reducing the learning rate to 0.01 didn't produce better results and neither incrementing it to 0.0001 (moving the gradient too slowly)
- Aumenting the `epochs` up to 25 didn't improve the predictions, resulting in a stagnant accuracy. This leaves me to think that I end up with a local minimum in the gradient regression or that the training set is not big enough.

General observations:
- Executing the whole training process more than once produced different final accuracies for the trained network. Due to randomized initial variables and weights, the execution of the training is not deterministic and the final result varies from an accurancy range of [0.87 - 0.925]
- The training accuracy is higher than the validation accuracy, indicating a slight over-fitting of the training model. Adding a layer of dropout `tf.nn.dropout(hidden_layer, keep_prob)` should help to prevent this effect in my current architecture


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] ![alt text][image2] ![alt text][image3] 
![alt text][image4] ![alt text][image5]

The second image is a non standard stop sign that has text inside the sign. It might be difficult to classify as the system has never seen text inside this kind of image. This sign is correctly indentified by the model.

The third image is a rotated warning sign with a dark background. The rotation and non uniformity of the image might not fit with any of the training set. The system misclassifies this image.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the ninth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Speed limit (30km/h)	| Speed limit (30km/h)							|
| No entry     			| No entry 										|
| General caution		| Road work										|
| Road work	      		| Road work					 					|
| Children crossing		| Children crossing     						|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80% (10th cell). This compares favorably to the accuracy on the test set of 87%

In the cell number 12, I have analysed the preformance of the model against the test set that only contains `General caution` traffic signs providing an accuracy of 0.818. This value is lower than the full test set accuracy of 0.879. This is problably a sign that incrementing the number of training data for this kind of signs would improve its accuracy

We could increase the training set by:
- Including rotations of the signs in all axis
- Adding sets with changing the background of the images
- Adding images with different light conditions
- Including images with normalised histogram
- Adding distorted images
- Including images not fully formed (as cropped or damaged)

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

I am not providing a char bar as the level of probability of the 4 predictions is too low to be appreciated.

For the first image Speed limit (30km/h)

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Speed limit (30km/h)   						|
| .2e-6    				| Speed limit (20km/h)							|
| .6e-8					| Speed limit (50km/h)							|
| .1e-11      			| Speed limit (70km/h)					 		|
| .2e-14			    | Speed limit (120km/h)      					|


For the second image No entry

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| No entry   									|
| .2e-4     			| Turn right ahead								|
| .3e-5					| Beware of ice/snow							|
| .1e-5	      			| Traffic signals					 			|
| .3e-8				    | Speed limit (100km/h)     					|

For the third image General caution but was predicted as a Road work, the correct prediction was the second most problable

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Road work   									|
| .9e-2    				| General caution								|
| .1e-8					| Pedestrians									|
| .5e-10	   			| Right-of-way at the next intersection			|
| .2e-10			    | Priority road  								|

For the forth image Road work

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .100         			| Road work   									|
| .1e-9    				| Bumpy road									|
| .5e-10				| Road narrows on the right						|
| .2e-10	   			| Dangerous curve to the right					|
| .2e-14			    | General caution   							|

For the fifth image Children crossing

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .99         			| Children crossing								|
| .1e-5    				| Bumpy road									|
| .6e-6					| Slippery road									|
| .6e-7	      			| Children crossing				 				|
| .5e-8				    | Road narrows on the right 					|
