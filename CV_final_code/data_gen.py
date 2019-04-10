import numpy as np
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing import image
import pathlib


class DataGenerator(keras.utils.Sequence):
    def __init__(self, img_paths, img_labels, feat_dict, batch_size=8, dim=(224,224,3), n_classes=50):
        self.img_paths = img_paths
        self.img_labels = img_labels
        self.feat_dict = feat_dict
        self.batch_size = batch_size
        self.dim = dim
        self.n_classes = n_classes
        self.indexes = np.arange(len(img_labels))

    def __len__(self):
        return int(np.floor(len(self.img_labels)/self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_paths_img = [self.img_paths[k] for k in indexes]
        batch_labels = [self.img_labels[k] for k in indexes]

        X, y = self.__data_generation(batch_paths_img, batch_labels)
        return X,y

    def __data_generation(self, batch_paths_img, batch_labels):
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)

        for i, idx in enumerate(batch_paths_img):
            img_tensor = image.load_img(idx, target_size=(224,224))
            img = image.img_to_array(img_tensor)
            img_final = img/255.0

            # lname = pathlib.Path(idx).parent.name
            # feat1 = self.feat_dict[lname][0]
            # feat2 = self.feat_dict[lname][1]
            #
            # concatenated = np.concatenate((img_final,feat1,feat2), axis=2)
            X[i,:,:,:] = img_final #concatenated
            y[i] = batch_labels[i]
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
