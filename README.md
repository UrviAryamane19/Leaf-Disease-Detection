Copyright 2019 Rahul J.Bhiwande , Urvi V.Aryamane , Chinmay A.Gandi , Saurabh V.Yadav

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

In this code, the model used is the " Inception V3 model " developed at Google, and pre-trained on ImageNet, a large dataset of images (1.4M images and 1000 classes). 
This is a powerful model; let's see what the features that it has learned can do for our apple_scab vs. apple_healthy.

To download the weights of the Inception model get it by running this command :
!wget --no-check-certificate \https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5
    
Also the dataset used in the code is available here :
https://drive.google.com/open?id=1V5LWEEHOqwWUs6sgJFE3IZaf4-8orfzX


1) Firstly , when u acess the code CNN.ipynb , it is a binary classifier , based on the architecture of Convolutional Neural Networks.
  It consists of 3 convolution layers , which contains 8 , 16 , 32 filters respectively for each layer of size 3 * 3.It is followed by
  flatten feature map to a 1-dim tensor so we can add fully connected layers.Then followed by fully connected layer with ReLU activation 
  and 512 hidden units.Then it consists of output layer with a single node and sigmoid activation.Due to the dataset being a bit scarce , this code doesn't resolve the issue of "Overfitting".
  
2) Now , CNN_overfitting.ipynb tries to resolve the previously mentioned issue of "overfitting".Usually a learning algorithm is trained using some set of "training data": exemplary situations for which the desired output is known. The goal is that the algorithm will also perform well on predicting the output when fed "validation data" that was not encountered during its training.Overfitting is the use of models or procedures that violate Occam's razor, for example by including more adjustable parameters than are ultimately optimal, or by using a more complicated approach than is ultimately optimal. For an example where there are too many adjustable parameters, consider a dataset where training data for y can be adequately predicted by a linear function of two dependent variables. Such a function requires only three parameters (the intercept and two slopes). Replacing this simple function with a new, more complex quadratic function, or with a new, more complex linear function on more than two dependent variables, carries a risk: Occam's razor implies that any given complex function is a priori less probable than any given simple function. If the new, more complicated function is selected instead of the simple function, and if there was not a large enough gain in training-data fit to offset the complexity increase, then the new complex function "overfits" the data, and the complex overfitted function will likely perform worse than the simpler function on validation data outside the training dataset, even though the complex function performed as well, or perhaps even better, on the training dataset.

Read more on : https://en.wikipedia.org/wiki/Overfitting

In this code we try to solve the problem by implementing data augmentation and the concept of dropout.

What augmentation will actually do here is , it will apply random no. of transformation on the image and will not make the model to see the same picture twice !!.This may lead to a reduction in overfitting as well as help us in generalize well.This is done by using ImageDataGenerator.

Few parameters of ImageDataGenerator used are :

   rotation_range is a value in degrees (0–180), a range within which to randomly rotate pictures.
   width_shift and height_shift are ranges (as a fraction of total width or height) within which to randomly translate pictures vertically                    
   or horizontally.
   shear_range is for randomly applying shearing transformations.
   zoom_range is for randomly zooming inside pictures.
   horizontal_flip is for randomly flipping half of the images horizontally. This is relevant when there are no assumptions of horizontal 
   asymmetry.
   fill_mode is the strategy used for filling in newly created pixels, which can appear after a rotation or a width/height shift.
    
Further datagen transformations are applied to produce five random variants.But only data augmentation is not enough to reduce overfitting
we need some more improvements.This introduces " dropouts ".This is done just after he fully connected layer.
By the statement  : Dropout(a)(x) , here a is the rate of dropout ,  x is the variable to which it is initialized.

To get a clear understanding of dropouts refer : https://www.commonlounge.com/discussion/694fd08c36994186a48d122e511f29d5

3) Finally the CNN_v3.ipynb , here as mentioned at the introduction the weights of Google's Inception v3 model have been used.In-depth explanation of the model is provided here : https://github.com/tensorflow/models/tree/master/research/inception. This model works well here as it has been trained on a large amount of dataset , this would in turn help in fine-tuning and feature extraction process.

Fine-Tuning :
    Fine-tuning should only be attempted after you have trained the top-level classifier with the pretrained model set to non-trainable. If       you add a randomly initialized classifier on top of a pretrained model and attempt to train all layers jointly, the magnitude of the gradient updates will be too large (due to the random weights from the classifier), and your pretrained model will just forget everything it has learned.
    Additionally, we fine-tune only the top layers of the pre-trained model rather than all layers of the pretrained model because, in a convnet, the higher up a layer is, the more specialized it is. The first few layers in a convnet learn very simple and generic features, which generalize to almost all types of images. But as you go higher up, the features are increasingly specific to the dataset that the model is trained on. The goal of fine-tuning is to adapt these specialized features to work with the new dataset.

Feature extraction :
    The layer used for feature extraction in Inception v3 is called mixed7. It is not the bottleneck of the network, but we are using it to keep a sufficiently large feature map (7x7 in this case). (Using the bottleneck layer would have resulting in a 3x3 feature map, which is a bit small.)
    
   Once you go through the code , you'll understand how the model has been seperated as firstly a non-trainable model for feature extraction and then for fine-tuning purpose varied to a trainable model with the exisiting weights.

Special thanks to  " Sanders Kleinfeld " for proper explanation of neural networks in a simple manner !
