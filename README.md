# UDACITY SDND - Semantic Segmentation Project Submission

[//]: # (Image References)
[image1]: ./lossprofiles.png
[image2]: ./um_000015.png
[image3]: ./um_000032.png
[image4]: ./um_000044.png
[image5]: ./um_000076.png
[image6]: ./umm_000038.png
[image7]: ./uu_000063.png
[image8]: ./uu_000064.png

## Introduction
In this project, the pixels of a road in images were labeled using a Fully Convolutional Network (FCN).

## APPROACH
A pre-trained VGG16 network was used as base and additional FCN layers were added with 1x1 convolutions and
upsampling combined with skip connections from previous layers. Upsampling was used to regain the image size.

## ARCHITECTURE
1. Pretrained VGG16 - 7 layer network was taken as baseline.
2. The Layer 7 output was 1x1 convolved and upsampled using TensorFlow `transpose()` function to match Layer 4 output size.
3. The output of step 2 was combined with '1x1 convolved Layer 4 output'
4. The output of step 3 was upsampled to match the size of Layer 3 output.
5. The output of step 4 was combined with  '1x1 convolved Layer 3 output'.
6. The result of step 5 was the last layer.
The logic was implemented in `main.py: layers()` method. This was implemented as per the paper `https://arxiv.org/pdf/1605.06211.pdf`.

## Loss and Optimizer
Cross Entropy Loss coupled with L2 Regularization loss.
Adam Optimizer was used.
The logic was implemented in `main.py:optimize()`.

## Training 
The following training were parameters used after some experimentation.
- Keep Probability = 0.5,  to avoid overfitting.
- learning_rate = 0.001
- epochs = 50
- batch_size = 5
The following picture captures the loss profiles with two scenarios.

![alt text][image1]

L2 regularization with slight increase in learning rate improved loss improvement pace.

## Results and Observations
The following are the sample images that in which road was labeled with green pixels. The result were robust.

![alt text][image2]

![alt text][image3]

![alt text][image4]

![alt text][image5]

![alt text][image6]

![alt text][image7]

![alt text][image8]



## The following sections belong to base readme.

### Setup
##### GPU
`main.py` will check to make sure you are using GPU - if you don't have a GPU on your system, you can use AWS or another cloud computing platform.
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [post](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/forum_archive/Semantic_Segmentation_advice.pdf) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
