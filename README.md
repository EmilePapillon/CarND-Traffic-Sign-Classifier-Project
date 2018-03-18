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

[image1]: ./images/bar-plot-trafficsigns.png "Visualization"
[image2]: ./images/normal.png "Grayscaling"
[image3]: ./images/trans_right.png "Random Noise"
[image4]: ./MyTestTrafficSigns/max70_roadbg.png "Traffic Sign 1"
[image5]: ./MyTestTrafficSigns/german_noentry_bluesky.png "Traffic Sign 2"
[image6]: ./MyTestTrafficSigns/stop.jpg "Traffic Sign 3"
[image7]: ./MyTestTrafficSigns/germany-speed-limit-sign.png "Traffic Sign 4"
[image8]: ./MyTestTrafficSigns/german-road-signs-animals.png "Traffic Sign 5"
[image9]: ./images/architecture.png "Architecture"
[image10]: ./images/loss_graph.png "Loss graph"
[image11]: ./images/Max_70_result.png "Max 70 Result"
[image12]: ./images/No_entry_result.png "No Entry Result"
[image13]: ./images/stop_result.png "Stop Result"
[image14]: ./images/max_60_result.png "Max 70 Result"
[image15]: ./images/wild_an_result.png "wild Animals Result"
[image16]: ./images/trans_left.png "trans left"
[image17]: ./images/trans_up.png "trans up"
[image18]: ./images/trans_down.png "trans down"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/EmilePapillon/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630 
* The shape of a traffic sign image is 32 by 32 pixels
* The number of unique classes/labels in the data set is 43

The sample german traffic signs from the web yielded an accuracy of 20% which is drastically different from the accuracy obtained from the test set.
#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed among the 43 different traffic sign types. The bar chart shows that some categories are underrepresented compared to some others. This can cause the CNN to not recognize some categories and to have a bias towards recognizing the ones that are overrepresented. 

To overcome this issue, it is necessary to augment the dataset, preferably only the categories that are underrepresented should be augmented while the others are left as-is.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The data was preprocessed using normalization and shuffling. Normalization was made by brigning each pixels - which are originally in the range 0-255 - to the range -1-1 by using the following transformation p = (p - 128) / 128. 

An attempt at augmenting the different bins was made but this did not generate better accuracy. The technique used was translating the images some amount of pixels to the right, left, top and bottom to generate 4 additional images for each image in the original bin. 

Then, iterating through the bins and counting their elements, augment the bins that don't have enough items according to a preset threshold (which was set to 1000). 

This technique did not improve the accuracy for unknown reasons but I suspect the following might explain why : 

*  The original dataset appears to have been augmented already. Augmenting an already augmented data set might generate too much redundancy in the data. 
*  Augmenting using translation might not be the best since convolutional network already are made for translation invariance since a patch is scanning the image using the stride as a translation unit. 

Here is an example of a traffic sign image before and after translating : 

![alt text][image2] ![alt text][image3] ![alt text][image16] ![alt text][image17] ![alt text][image18]

I also tried other tools to augment my data set. I used the imgaug library and applied blending / overlaying images, cropping, flipping and gaussian blur combinations to generate random images for augmenting the data set. I used this only on the bins that had too few item. The effect of this was to decrease my validation accuracy, so I gave up augmentation . 

As a last step, I normalized the image data because the backpropagation algorithm performs better when the data is well conditionned. This means the mean of the pixels must be zero and the variance must be constant. A goot way to normalize is to bring every pixel value between -1 and 1. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used the LeNet architecture as a basis for my network. I modified the original architecture by adding dropout layers after the first and the second convolutional layers. I added the dropout before and after the max pooling and tested different configurations. I found that the configuration that performed the best was when I used the dropout before the max-pooling. This makes sense because the max-pooling is already rejecting outputs. If dropout is aplied to the result of the rejection, this makes too much information loss. But if instead dropout is used before, the max-pooling takes the maximum between 0s generated by the dropout and values of the logits that were unafected by the dropout operation and less information is lost as a result. 

![alt text][image9]  

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For the training, I used the Adam optimizer to minimize the mean cross-entropy. The original batch size was reduced from 128 to 32, as this increased the validation accuracy. 

The network was trained for a total of 35 epochs using a lower learning rate of 0.0005 as this optimized the validation accuracy. After epoch 15 though there is no more improvement in the validation accuracy as the network tends to overfit. As discussed earler, a better augmented dataset could have solved this problem. 

![alt text][image10]

The training loss - validation loss graph above shows that is was not necessary to train for 35 epochs as the improvement of the validation accuracy is almost flat after 15 epochs of training.  

At epoch 33 the validation accuracy was maximal at 94.7%. 


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with the LeNet architecture as-is and it performed at between 87 and 89 % accuracy. I noticed that the dataset was not well balanced and tried translation to augment it which didn't yield satisfactory results. After plotting the validation loss against the trainig loss, I concluded that my network was suffering from overfitting. Dropout was one of the methods explained during class to overcome overfitting problems but I didn't know if I should put the dropout before or after max-pooling. Some experimenting led me to discover that it performed better when but before the max-pooling. After using dropout and lowering the batch size and learning rate, I got results in the range 93 - 95 % in accuracy.

My final model results were:
* training set accuracy of  98.5% 
* validation set accuracy of 93% 
* test set accuracy of 93%

Choosing the LeNet architecture made sense because of the known performance of this architecture for this type of problems. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the traffic sign is small relatively to the size of the image. 

The only image that was sucessfully recognized was the last one showing wild animals crossing. I am surprized that my model did not perform any better on the penultimate image which was very similar to the training set. Normalization and resizing were applied to all images. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

The model is sure about all the images but is failing to make the correct precidtion 4 times out of 5, which is very bad. One reason might be it has overfitted to the training set althought the validation accuracy indicates otherwise. Another possibility might be the dataset is very different from the images shown, but that doesn't seem to be the case at lease for the maximum 60 kmh image and the animals crossing, which is correctly guessed. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| keep left   									| 
| 0.0     				| Beware of ice/snow 										|
| 0.0					| Speed limit 120 km/h											|
| 0.0	      			| Speed limit 80 km/h					 				|
| 0,0			    | Animals crossing      							|


For the second image which is a no entry sign, the prediction is also wrong : 

| Probability           |     Prediction                                                        | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                           | priority road                                                                    |
| 0.0                                   | Right of way at the next intersection                                                                            |
| 0.0                                   | Yield                                                                                  |
| 0.0                           | Traffic signals                                                                   |
| 0,0                       | Speed limit 20 km/h                                                          |

For the third image which is a stop sign, the prediction is also wrong: 

| Probability           |     Prediction                                                        | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                           | priority road                                                                     |
| 0.0                                   | No entry                                                                            |
| 0.0                                   | No passing                                                                                  |
| 0.0                           | No passing for vehicles over 3.5 metric tons                                                                   |
| 0,0                       | Road work   

                                                       |
The fourth image is a maximum 60 km/h and the prediction is wrong although the model seems to have picked up the 'speed limit' concept. 

| Probability           |     Prediction                                                        | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                           | Speed limit 50 km/h                                                                     |
| 0.0                                   | Speed limit 30 km/h                                                                            |
| 0.0                                   | Speed limit 80 km/h                                                                                  |
| 0.0                           | Speed limit 60 km/h                                                                   |
| 0,0                       | end of speed limit 80 km/h                                                          |

The last one is the only one that is accurately predicted by the model : 

| Probability           |     Prediction                                                        | 
|:---------------------:|:---------------------------------------------:| 
| 1.0                           | Animals crossing                                                                     |
| 0.0                                   | Speed limit 20 km/h                                                                            |
| 0.0                                   | Speed limit 30 km/h                                                                                  |
| 0.0                           | Speed limit 50 km/h                                                                   |
| 0,0                       | Speed limit 60 km/h                                                          |

There are many "speed limit" predictions, which might be due to the fact the dataset is biased by having more of these examples and less of the others. I have yet to find out a good way to augment it. Strangely there is a huge difference between the performance of those images and the accurancy of the test set.  
