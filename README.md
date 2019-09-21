# Generate-Shoe-Designs

This repository is developed using the concept of Generative Adversarial Networks(DCGAN & WGAN) to design and generate new images of shoes.

Dataset: The large dataset of shoes consisting of 50,025 images can be downloaded from Zappos.com
The images are divided into 4 major categories — shoes, sandals, slippers, and boots. The shoes are centered on a white background and pictured in the same orientation for convenient analysis.
The link to the website containing dataset is: http://vision.cs.utexas.edu/projects/finegrained/utzap50k/

Methods: For the Generator, a series consisting of strided 2D Convolutional Transpose layers paired with a 2D Batch Normalization and a ReLU activation is used. Output is fed through a tanh Activation function to return it to the input data range [-1, 1]. 
For the Discriminator, a series consisting of strided 2D Convolutional layers paired with a 2D Batch Normalization and a Leaky ReLU activation is used. Linear Activation is used for calculating the final probability.

WGAN replaces the discriminator model with a critic to check if the given image is real or fake. The implementation of a WGAN requires a few changes to the DCGAN.
1. Use a linear activation function in the output layer os the discriminator model instead of the sigmoid.
2. Use Wasserstein loss to train the discriminator & generator models tht promotes larger differences between scores for real and fake images.
3. Constrain discriminator model weights to a limited range after each mini batch update.

Dependencies: python-3, tensorflow-1.14, numpy, PIL, scipy

Usage: python3 models.py
