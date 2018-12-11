# **Traffic Sign Recognition**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/BefViz.png "Visualization"
[image2]: ./examples/GrayNorm.png "Grayscaling+Normalization result"
[image3]: ./examples/Resized.png "Resize"
[image4]: ./RoadSigns/ID3.png "Traffic Sign 1"
[image5]: ./RoadSigns/ID11.png "Traffic Sign 2"
[image6]: ./RoadSigns/ID12.png "Traffic Sign 3"
[image7]: ./RoadSigns/ID18.png "Traffic Sign 4"
[image8]: ./RoadSigns/ID25.png "Traffic Sign 5"
[image9]: ./examples/AftViz.png "Visualization after augmenting data"
[image10]: ./RoadSigns/ID38.png "Traffic Sign 6"
[image11]: ./examples/Warped.png "Warp"
[image12]: ./examples/Bright.png "Brightness Change"
[image13]: ./examples/Translated.png "Translate"
---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 samples
* The size of the validation set is 4,410 samples
* The size of test set is 12,630 samples
* The shape of a traffic sign image is (32,32,3)(width(pixels),height(pixels),color channels(RGB))
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a bar chart showing how the training data is distribuited across labels:

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it would allow me to reduce the number of inputs into the model and it would make normalization easier.

Here is an example of a traffic sign image before and after grayscaling and normalization:

![alt text][image2]

As a last step, I normalized the image data by pixel=(pixel-128)/128 because that will make the mean of all input values(pixels) closer to 0, since before they ranged from 0 to 255.

I decided to generate additional data because some labels were lacking examples and I wanted the model to be exposed to a good amount of each type of sign during training.

To add more data to the the data set, I used the following techniques because it was easy to implement and would build up the labels lacking examples so the least amount of samples for a label in training data would be 900 and a good amount of the labels would end up with similar amounts of samples in the training set

Here is an example of the 4 images added to the training data if the original image's label had less than 1000 samples:

Slight Random Resize Along Width and Height:
![alt text][image3]

Slight Random Warp:
![alt text][image11]

Slight Random Brightness Change:
![alt text][image12]

Slight Random Translation Along Width and Height(pixels that would be translated out of image are added to other side):
![alt text][image13]


The difference between the original data set and the augmented data set is the following all labels that had less than 1000 samples in the training set now have 5 times as many making the smallest number of samples for a label be 900.

Here is a bar chart showing how the training data is distribuited across labels after data augmentation:
![alt text][image9]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 Grayscale image   					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,same padding,outputs 14x14x6 		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,same padding,outputs 5x5x16 		|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					|												|
| Flattten 				| 1x1x400 to 400        						|
| Flattten Layer1 output| 14x14x46 to 1176        						|
| Concatenate 			| 400+1176 to 1576        						|
| dropout				| keep_prob=.5        							|
| Fully connected		| 1576 to 43        							|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 128 over 20 epochs with a learning rate of .00095 and the adamoptimizer trying to minimize the output of the loss calculation, tf.reduce_mean.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of .952
* test set accuracy of .948

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    LeNet, I thought might well try it to see how well it works since I already had the architecture.

* What were some problems with the initial architecture?
    It got up to .952(coincidentlly same as the architecture I currently have) on validation, but the accuracy started going up and down after just a few epochs.(happening by epoch 7) so I decided to try tweaking it.  

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    I believe I had underfitting, since validation accuracy was going up and down so I gradually changed it, to a model architecture I found online that was also worked on by Yann LeCun. [The paper I found](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), said they were able to get high accuracy using the model as well as by augmenting the data set used to train the model. Before I ended up making the model the same general architecture as the one in the paper I tried various things, such as leaving in the multiple fully connected layers from LeNet after the paper's model, adding dropout, and adding/removing maxpooling in places. Ultimately, I found I would do well on the test and validation sets but was getting 0 out of 6 images right on my images with low probabilities and the same top 5 labels for all 6 images, so I just implemented it as the model structure from the paper and still had this problem leading me to believe that the way I augmented the training set made it bad to train on, or that maybe I should concatenate the 2nd convolutional layer's output with the 3rds before feeding into the fully connected layer since it might be detecting higherlevel features in the image than the first that are more indicative of and unique to certain labels.

* Which parameters were tuned? How were they adjusted and why?
    Epochs to give it more time to try and gradually minimize the loss function(I was going to try and add learning rate decay since validation accuracy was going up and down at times, but I found online that the Adamoptimizer already implements this).

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
    [A paper on the architecture I ended up trying to implement](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
* Why did you believe it would be relevant to the traffic sign application?
    The paper I found stated that this was the model's purpose

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
     The validation and test accuracy's are high, but on my images it was not confident with its classifications got 1 right(by luck?)

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image10]

The image of the 60km/h sign might be difficult to classify since it has part of another sign above it. Since the image of the priority road sign has an all white background it might be different from the types of backgrounds found for signs in the training set making it difficult to classify, I think the same might be true for some of the other signs as their backgrounds are pretty solid colors like the clear blue sky, or a plain white wall.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			                |     Prediction	        			|
|:-----------------------------:|:-------------------------------------:|
| Right-of-way next intersection| Road Work								|
| 60km/h    					| Road Work 							|
| Priority road					| Road Work								|
| General caution				| Road Work					 			|
| Road work						| Road Work 							|
| Keep right					| Road Work 							|


The model was able to correctly guess 1 of the 6 traffic signs, which gives an accuracy of .167. This compares is very poor compared to .948 on the test set.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For all the images it pedicted the same 5 signs with the highest probability for any sign being .478, all had Road work as their top probability, and the order of the other 4 had similar probabilities, .05-.15, except for the keep right which had keep right(the correct label) as its second highest probability with .25

Actual:Road Work
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .478         			| Road Work   									|
| .159     				| Speed limit (80km/h) 							|
| .115					| Beware of ice/snow							|
| .062      			| Keep right 			 						|
| .062				    | No passing									|

Actual:Keep Right
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .329         			| Road Work   									|
| .253     				| Keep Right 									|
| .134					| Beware of ice/snow							|
| .132      			| Speed limit (80km/h)			 				|
| .045				    | No passing									|

Actual:Right-of-way next intersection
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .455         			| Road Work   									|
| .168     				| Beware of ice/snow 							|
| .137					| Speed limit (80km/h)							|
| .071      			| Keep right 			 						|
| .052				    | No passing									|

Actual:Priority road
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .420         			| Road Work   									|
| .159     				| Speed limit (80km/h) 							|
| .125					| Beware of ice/snow							|
| .083      			| Keep right 			 						|
| .077				    | No passing									|

Actual:General caution
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .473         			| Road Work   									|
| .145     				| Beware of ice/snow 							|
| .137					| Speed limit (80km/h)							|
| .081      			| Keep right 			 						|
| .048				    | No passing									|

Actual:60km/h
| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .427         			| Road Work   									|
| .182     				| Speed limit (80km/h) 							|
| .104					| Beware of ice/snow							|
| .086      			| Keep right 			 						|
| .057				    | No passing									|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Plan on adding this after figure out what is wrong with rest of my process(data/data augmentation or model) and fix it.
