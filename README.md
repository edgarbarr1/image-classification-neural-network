# X-Ray Image Classification Network using Convolutional Neural Network #

Using Neural Networks to predict pneumonia based on X-ray images of patient's lungs.

![image1](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/doc_holding_xray.jpeg)

## Repository Directory ##

```
├── README.md        <-- Main README file explaining the project's business case,
│                        methodology, and findings
│
│
├── notebook         <-- Jupyter Notebooks for exploration and presentation
│
├── src              <-- Data dictionaries, manuals, and project instructions
│
└── images           <-- Generated graphics and figures to be used in reporting
```


# Background #

It has been a long feverish dream of science nerds for computers to have the sentient intelligence to be able to make intelligent decisions based on minimal data. With **Convolutional Neural Networks** that dream is one come true. At least, to some extent. 

The **purpose** of this project is to show the an iterative process of building Neural Networks as we slowly start going *deeper* in learning and evaluating the model's performance. 

_**Disclaimer:**_ Exact model perfomance may vary when run. The last few models may take a more than a few minutes to run.

# Data #

The data used in this repository comes from the [Mendeley](https://data.mendeley.com/datasets/rscbjbr9sj/3). The specific data used in this dataset was the **chest_xray** directory which contained separate folders for **train** and **test** images. Both the **train** and **test** directories had separate directories for the 2 categories of images, in this case *NORMAL* lung X-rays and *PNEUMONIA* lung X-rays. 

The **train** directory as a whole had over 5000 images for both categories and was train-test-splitted to fit our Neural Network. the **test** directory held over 600 images for both categories and was used as our final validation data for our models.

For this project, it was assumed that all images were correctly categorized in their respective categories. No additional research was done to verify the accuracy of the categorized images.

**_Sources:_** Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images”, Mendeley Data, V3, doi: 10.17632/rscbjbr9sj.3

# Preprocessing #

In order to get our images ready for our models our images were added to a specified array that included the the array information of the images and its label. The train_data was the target array used to train our models. The test_data was the holdout data used to validate our last couple of models.

Afterwords, the train and test dataset's were split into the image arrays and labels (X, y). The data was then normalized by dividing the image arrays by 255. 

The X and y variables were then train-test-split, using 25% for testing purposes and 75% as the training data. This gave 3924 images to train our models and 1308 images to test.

Our holdout data had 624 images to validate our models.

# Models #

A total of 4 Convolutional Neural Networks were created to try to pass our data into. Below is the overview of the models and it's graphical performace.

## Baseline Model ##

The first model built was a baseline model and not a lot of layers were added. This model was created to see the model performance at a first glance before going deeper in layers. See below for model performance.

![image_2](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/baseline_model.png)

Overall the model had good scores with some overfitting starting to happen after the 4th Epoch. As seen above, the recall score for this model was a 97.02%. This model ran through the data in a little more than a minute in a half.

## Model 2 ##

For our second model, it was decided that we were still not going to go very deep in layers. The main additions to the model was one additional Dense layer with a relu activation formula and a higher amount of filters. See below for model performance.

![image_3](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/model_2.png)

This model had a slightly better recall score (97.17%) than the previous however the overfitting problem still remained by a point. This model ran through the data in just over 2 minutes.

## Model 3 ##

In this model we decided to add more layers and bigger filters to those layers. To begin with, a Dropout layer was added to try and regularize the data and prevent some overfitting. Then a couple more Dense layers were added to the data. See below for model performance.

![image_4](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/model_3.png)

After running our model through 10 epochs, our model performed a solid 95.39% recall score. Although a lower score compared to our 2 prior models, there was significantly less overfitting on our training data. This model ran through our data in 12 minutes.

After validating our model with our holdout data, we had an overall recall score for this model of 88.30%. Although there was a drop in model performance when using new data, it still passed the 85% goal that we had in the beginnning.

## Model 4 ##

This model had significantly more layers. This model had 2 Dropout layers and 2 more Conv2D layers along with the pooling layers. See below for model performance.

![image_5](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/model_4.png)

After running through 15 epochs, this model performed at a respectable 97.63% recall and had a bit of overfitting in the training data. After running it through our validation data, this model performed at a 84.78% recall. Keeping this score in mind and that the model's runtime was 30 minutes, it is safe to conclude that this was not our best performing model despite its depth in layers.

# Explaining the Best Performing Model #

With a validated score of. 88.30% recall, and a runtime of 12 minutes, the best perfoming model was the third model. Using this model and a package named [lime](https://github.com/marcotcr/lime), we were able to "see" what the computer actually saw in 3 pictures when "deciding" what label was each picture. With a little bit of help of the creator of this package - [marcotcr](https://github.com/marcotcr) - we were able to fit our model and a 3 pictures - each from the different X value datasets _ and visualize the model's learning.

## Image 1 ##

![first_image](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/first_image.png)

The first image came from the X_train dataset and had a (1,0) label which is for NORMAL X-ray. After verifying that the image was ready to be used on our model to predict we passed it through and these were the results.

Segmented Full Image | Segmented Partial Image
------------ | -------------
![first_image_unblackened](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/first_image_top_pixels_unblackened.png) | ![first_image_blackened](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/first_image_top_pixels_blackened.png)
Pros and Cons | Heatmap
------------- | -------------

![first_image_pros_cons](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/first_image_pros_cons.png) | ![first_image_heatmap](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/first_image_heatmap.png)column

**Segmented Full Image**

![first_image_unblackened](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/first_image_top_pixels_unblackened.png)

**Segmented Partial Image**

![first_image_blackened](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/first_image_top_pixels_blackened.png)

**Pros and Cons**

![first_image_pros_cons](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/first_image_pros_cons.png)

**Heatmap**

![first_image_heatmap](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/first_image_heatmap.png)

## Image 2 ##

![second_image](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/second_image.png)

The second image came from the X_test dataset and had a (1,0) label for NORMAL. After verifying that the image was ready to be used on our model to predict we passed it through and these were the results.

**Segmented Full Image**

![second_image_unblackened](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/second_image_top_pixels_unblackened.png)

**Segmented Partial Image**

![second_image_blackened](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/second_image_top_pixels_blackened.png)

**Pros and Cons**

![second_image_pros_cons](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/second_image_pros_cons.png)

**Heatmap**

![second_image_heatmap](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/second_image_heatmap.png)

## Image 3 ##

![third_image](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/third_image.png)

The second image came from the Xtest dataset and had a (1,0) label for NORMAL. After verifying that the image was ready to be used on our model to predict we passed it through and these were the results.

**Segmented Full Image**

![third_image_unblackened](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/third_image_top_pixels_unblackened.png)

**Segmented Partial Image**

![third_image_blackened](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/third_image_top_pixels_blackened.png)

**Pros and Cons**

![third_image_pros_cons](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/third_image_pros_cons.png)

**Heatmap**

![third_image_heatmap](https://github.com/edgarbarr1/image-classification-neural-network/blob/main/images/third_image_heatmap.png)

# Conclusion #

After evaluating all of our models, we can say that our best performing model is model 3. Although our runtime was significantly longer than our first 2 models, there was no significant overifitting compared to our first 2 models. Hence choosing model 3 as our best performing mode1.

A few things that we could do for future tuning is verify the accuracy of the images in the different directories of the original datasets. That could be a reason for the drop of recall score when validating the model. 
