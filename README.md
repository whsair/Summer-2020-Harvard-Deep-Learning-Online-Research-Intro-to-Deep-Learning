# Harvard Deep Learning Online Research

- **Instructor: Harvard Lecturer. Pavlos Protopapas**
- **Date: 06/2020 - 08/2020**
## Overview:
In this session, you will learn about artificial neural networks and how they're being used for machine learning. We'll emphasize both the basic algorithms and the practical tricks needed to get them to work well. The objective of this session is to provide students with an intuition and understanding of artificial neural networks including subjects such as architecture choices, activation functions, feed-forward, convolutional neural networks and auto-encoders.

## Research
### Introduction to NN, review of classification and regression, optimization, backpropagation and simple Feed Forward (FF) Network
- [The Perceptron](Lecture%201%20Ex1%20-%20The%20Perceptron)
- [Multilayer Perceptron with Keras](Lecture%201%20Ex2%20-%20Multilayer%20Perceptron%20with%20Keras)
### Neural Network Architecture, Design Choices
- [NN Architecture](Lecture%202%20Ex1%20-%20NN%20Architecture)
- [Multi-class Classification](Lecture%202%20Ex2%20-%20Multi-class%20Classification)
### Regularization for Neural Networks, dropout, and batch normalization
- [Gradient Descent](Lecture%203%20Ex1%20-%20Gradient%20Descent)
- [Decay & Clipping](Lecture%203%20Ex2%20-%20Decay%20%26%20Clipping)
- [Momentum](Lecture%203%20Ex3%20-%20Momentum)
- Batch Normalization
### Backpropagation and Optimizers
- [Adam](Lecture%204%20Ex1%20-%20Adam)
- [Weight Decay](Lecture%204%20Ex2%20-%20Weight%20Decay)
- [Early Stopping](Lecture%204%20Ex3%20-%20Early%20Stopping)
- [Dropout](Lecture%204%20Ex4%20-%20Dropout)
### Convolutional Neural Networks: basic concepts and architectures
- Feed-Forward Neural Network & Convolutional Neural Networks
### Convolutional Neural Networks: receptive fields, strides, etc.
- Padding, stride and max pooling
### Convolutional Neural Networks: neural net transfer learning and state of the art networks
- Training CNNs
- BackProp of MaxPooling layer
- Layers Receptive Field
- Saliency maps
- Transfer Learning
### Auto-encoders
- [(Dense) Autoencoder on MNIST](Lecture%208%20Ex0%20-%20(Dense)%20Autoencoder%20on%20MNIST)
- [Autoencoder on Iris data](Lecture%208%20Ex1%20-%20Autoencoder%20on%20Iris%20data)
- [ConvAutoencoder Olivetti Faces](Lecture%208%20Ex2%20-%20ConvAE%20Olivetti%20Faces)

### [Project 1 ANNs and Model Interpretability](HW1)
#### Construct a feed forward neural network
- In this part of the homework, you are to construct three feed-forward neural networks. Each neural network will consist of 2 hidden layers and an output layer. The three different networks only differ in their number of nodes used for their hidden layer, which we specify in each specific question below. All networks' hidden layers use the sigmoid as the activation function, along with a linear output node.
#### Neural Networks
- Neural networks are, of course, a large and complex topic that cannot be covered in a single homework. Here, we'll focus on the key idea of ANNs: they are able to learn a mapping from example input data X (of fixed size) to example output data Y (of fixed size). This is the same concept as every other classification and regression task we've learned so far in the semester. We'll also partially explore what patterns the neural network learns and how well neural networks generalize.

- In this question, we'll see if neural networks can learn a limited version of the Fourier Transform. (The Fourier Transform takes in values from some function and returns a set of sine and cosine functions which, when added together, approximate the original function.)
#### Regularizing Neural Networks
- For this problem, we will be working with a modified version of MNIST dataset (MNIST CS109, MNIST: Modified National Institute of Standards and Technology database), which is a large database of handwritten digits and commonly used for training various image processing systems. This dataset consists of 60,000 28x28 grayscale images of the ten digits, along with a test set of 10,000 images. For pedagogical simplicity, we will only use the digits labeled 4 and 9, and we want to use a total of 1600 samples for training (this includes the data you will use for validation).

- We have selected the samples for you and the dataset is available at https://www.kaggle.com/c/intro-to-nns-hw1-june-2020/data. You have to create an account on Kaggle and join the competition via https://www.kaggle.com/t/ec2eba573312496d8f53a8ec8f18695d. This is a limited participation competition. Please do not share link. Note, it's not technically a competition, as your goal is merely to create an appropriate, strong model that performs well. We will evaluate your skills based on this, not in terms of how you compare to your classmates.
<img src="https://github.com/whsair/Summer-2020-Harvard-Deep-Learning-Online-Research-Intro-to-Deep-Learning/blob/main/HW1_result2.PNG" width="400" /> | <img src="https://github.com/whsair/Summer-2020-Harvard-Deep-Learning-Online-Research-Intro-to-Deep-Learning/blob/main/HW1_result1.PNG" width="400" />

### [Project2 CNNs](HW2)
#### Convolutional Neural Network Mechanics
As you know from lecture, in convolutional neural networks, a convolution is a multiplicative operation on a local region of values. Convolutional layers have shown themselves to have been very useful in image classification, as they allows the network to retain local spatial information for feature extraction.
#### CNNs at Work
- Load the image as a 2D Numpy array into the variable library_image_data. Normalize the image data so that values within library_image_data fall within [0., 1.]. The image is located at 'data/Widener_Library.jpg'.
#### Building a Basic CNN Model
- In this question, you will use Keras to create a convolutional neural network for predicting the type of object shown in images from the CIFAR-10 dataset, which contains 50,000 32x32 training images and 10,000 test images of the same size, with a total of 10 sizes.

- Loading CIFAR-10 and Constructing the Model.

- Load CIFAR-10 and use a combination of the following layers: Conv2D, MaxPooling2D, Dense, Dropout and Flatten Layers (not necessarily in this order, and you can use as many layers as you'd like) to build your classification model. You may use an existing architecture like AlexNet or VGG16, or create one of your own design. However, you should construct the network yourself and not use a pre-written implementation. At least one of your Conv2D layers should have at least 9 filters to be able to do question 3.3.

#### Image Orientation Estimation
- In this problem we will construct a neural network to predict how far a face is from being "upright". Image orientation estimation with convolutional networks was first implemented in 2015 by Fischer, Dosovitskiy, and Brox in a paper titled "Image Orientation Estimation with Convolutional Networks", where the authors trained a network to straighten a wide variety of images using the Microsoft COCO dataset. In order to have a reasonable training time for a homework, we will be working on a subset of the problem where we just straighten images of faces. To do this, we will be using the CelebA dataset of celebrity faces, where we assume that professional photographers have taken level pictures. The training will be supervised, with a rotated image (up to 60 deg) as an input, and the amount (in degrees) that the image has been rotated as a target.

## [Final Report: MobileNet_V2](Final%20Report)
- **Depthwise Separable convolution**
- **Structure** 
- **Relu6**
- **Inverted Residuals**
- **Linear Bottlenecks**
- **Transfer Learning**
    - Training datasets: the **sign language gestures** for the digits 0-9

    - Modified last layers from 1000 class to 10 (10 signs in total)

    - Freeze the first half of trained model and only train the last half (# train 10-16th bottle blocks

- **Visualization**
##### Results: [_Report ppt_](Final%20Report/Mobilenet%20v2.pptx) | [_Report video_](https://youtu.be/UTKTGsrbhmY)

