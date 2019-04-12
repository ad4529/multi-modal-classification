# Augmenting Image Classification with Categorical/Sparse Features

In this project, it was shown that augmenting a CNN model with extra features such as a short bio and a collection of categorical features such as age, active years,  nationality etc along with image data can have a positive impact on the models' prediction performance. The concept has been tested on the [best-artworks-of-all-time](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) dataset - a Kaggle dataset containing about ~8500 images of famous paintings of 50 greatest artists of medieval times. In addition to the images, the dataset also has features specific to each artist - categorical and sparse features which have been used to augment each testing, validation and training image. The figure below shows the entire pipeline that has been used to augment performance.

![Network Pipeline](fig1.jpg?raw=True "Training Strategy")

## Requirements

Although other combinations of the below libraries should work, the code has been developed with:
* Tensorflow 1.13 with CUDA 10 and CuDNN 7.5.1
* Keras 2.2.4
* Gensim Doc2Vec
* nltk

## Results

Both VGG19 and ResNet152 were used to test this concept. While the results on ResNet is still in the works, on VGG19, two ablation experiments were performed to demonstrate the efficacy of the proposed method - without feature augmentation and with feature augmentation. It was seen that mean squared error loss worked best for only images and categorical crossentropy for the augmented input. The base model gave a top-1 test acuracy of `91.4%` while the augmented model increased the top-1 performance to `~97%`. This considerable jump in performance can be attributed to the network learning to make sense of the extra features along with the RGB channels. Some pre-processing and normalizing of feature and image data had to done to ensure performance can improved including clipping the vector representations between `0 and 1`.

## How to run

### Training

Comment out the prediction part of the code in `load_data.py'.

### Testing

Comment out the `fit_generator()` line in `load_data.py` after you have your saved model.
