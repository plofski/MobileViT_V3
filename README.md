# MobileViT3XXS implementation in keras.
This is an implementation in keras of the MobileViT3XXS architecture. I changed the code of the original Keras code (found [here](https://keras.io/examples/vision/mobilevit/)) so that it complies with the changes made 
in the paper "MobileViTv3: Mobile-Friendly Vision Transformer with Simple and Effective Fusion of Local, Global and Input Features" [here](https://arxiv.org/abs/2209.15159)
I also train the MobileViTv1 and MobileViTv3 to the CIFAR10 dataset to compare what the advantages are of my implementation of  MobileViTv3 compared to MobileViTv1.
I also added the best weights of the mobileViTv3 model for CIFAR10 that got around 83.86% of accuracy on the test set. 

# What did I find?
It seems that the MobileViTv3 is less prone to overfitting (I think due to the extra skip connections) than MobileViTv1. 
