# Augmenting Image Classification with Categorical/Sparse Features

In this project, it was shown that augmenting a CNN model with extra features such as a short bio and a collection of categorical features such as age, active years,  nationality etc along with image data can have a positive impact on the models' prediction performance. The concept has been tested on the [best-artworks-of-all-time](https://www.kaggle.com/ikarus777/best-artworks-of-all-time) dataset - a Kaggle dataset containing about ~8500 images of famous paintings of 50 greatest artists of medieval times. In addition to the images, the dataset also has features specific to each artist - categorical and sparse features which have been used to augment each testing, validation and training image. The figure below shows the entire pipeline that has been used to augment performance.

![Network Pipeline](fig1.jpg?raw=True "Training Strategy")
