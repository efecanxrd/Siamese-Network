<h2> Siamese Network </h2>
<h3> Siamese neural network using Keras, which compares the similarity of two images from MNIST dataset and trains the model using contrastive loss function. </h3>
<img src="https://i.imgur.com/qHAcfhX.gif">
<h3> Setup The Project </h3>
<h4><img align="center" src="https://raw.githubusercontent.com/efecanxrd/efecanxrd/main/images/xe.gif" width="30"> Install Python <h4>
<h5>I recommend that you install Python3x </h5>
<h4><img align="center" src="https://raw.githubusercontent.com/efecanxrd/efecanxrd/main/images/xe.gif" width="30"> Install Libraries </h4>
<h5> You can install the library by typing 'pip install numpy' and 'pip install keras' in the terminal </h5>
<h2> How this is working? </h2>
<h4> The code creates a Siamese neural network that compares the similarity between two images from MNIST dataset. The network is implemented using Keras library in Python.

The network is made up of two identical sub-networks, each of which takes an image as input. The output of these sub-networks is then passed through a distance metric (Euclidean distance) which calculates the distance between the two outputs. A loss function called contrastive loss is used to train the model. The network is trained on pairs of images, where the goal is to minimize the contrastive loss so that the network can learn to distinguish similar images from dissimilar ones.

The code uses the MNIST dataset, which is a dataset of handwritten digits. The dataset is loaded and the images are preprocessed by normalizing the pixel values between 0 and 1. The code then creates pairs of images, both positive and negative, where positive pairs are images of the same digit and negative pairs are images of different digits. The pairs are then used to train the siamese network.

The code uses several Keras functions and layers such as Model, Input, Flatten, Dense, Dropout, Lambda, RMSprop to define the architecture of the network, the loss function and the optimizer.

 </h4>
<h4> You can get information by the link below. You might need to translate the page <h4>

[<img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white"/>](https://efecanxrd.medium.com/siyam-sinir-a%C4%9Flar%C4%B1-siamese-neural-network-e5413ee121)
  
