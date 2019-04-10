import glob
import tensorflow as tf
import pathlib
import random
from sklearn.model_selection import train_test_split
import keras
from keras.callbacks import TensorBoard, LambdaCallback
from keras.layers import Conv2D, Softmax, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense, Activation, BatchNormalization, InputLayer, GaussianNoise
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras import regularizers
from data_gen import DataGenerator
import cv2
import matplotlib.pyplot as plt
import numpy as np
import functools
from cat_parse import get_vector_from_cats
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


data_root = '/home/ad0915/Desktop/CVFinalDataset/best-artworks-of-all-time/images'
data_root = pathlib.Path(data_root)


all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

labels = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
labels_to_idx = dict((name, idx) for idx, name in enumerate(labels))

all_image_labels = [labels_to_idx[pathlib.Path(path).parent.name] for path in all_image_paths]

train_val_imgs, test_imgs, train_val_labels, test_labels = train_test_split(all_image_paths, all_image_labels, test_size=0.10, random_state=21)
train_imgs, val_imgs, train_labels, val_labels = train_test_split(train_val_imgs, train_val_labels, test_size=0.1111, random_state=21)

base_model = keras.applications.vgg19.VGG19(weights=None, include_top=False, input_shape=(224,224,5))

inter = Sequential()
inter.add(base_model)
#inter.add(MaxPooling2D(pool_size=(7,7)))
inter.add(Flatten())
inter.add(Dropout(0.3))
inter.add(Dense(1024, kernel_regularizer=regularizers.l2(0.01)))
inter.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
inter.add(Dense(50, activation='softmax'))
# x = MaxPooling2D(pool_size=(2,2))(x)
# x = Flatten()(x)
# x = Dense(1000)(x)
# x = Dense(500)(x)
# preds = Dense(50, activation='softmax')(x)
#
# final_model = Model(inputs=noise.input, outputs=preds)

sgd = SGD(lr=0.001, decay=0.00004)
top3_acc = functools.partial(metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
inter.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', top3_acc])

inter.summary()

feat_dict = get_vector_from_cats()

training_generator = DataGenerator(train_imgs, train_labels, feat_dict)
validation_generator = DataGenerator(val_imgs, val_labels, feat_dict)
testing_generator = DataGenerator(test_imgs, test_labels, feat_dict)

cp_callback = ModelCheckpoint('/home/ad0915/Desktop/CV_final/VGG19_weights/aug_bce_rms.h5', monitor='val_acc', verbose=1, save_best_only=True, period=4)

#inter.fit_generator(generator=training_generator, validation_data=validation_generator, validation_steps=int(np.floor(850/8)), epochs=400, use_multiprocessing=True, workers=8, callbacks=[cp_callback])

model = load_model('./VGG19_weights/base_msle_sgd0.03_all_noise.h5', custom_objects={'top3_acc':top3_acc})
model.summary()
loss, acc, top3_acc = model.evaluate_generator(testing_generator, verbose=1)
print(' Testing Accuracy = {:5.2f}%'.format(100*acc))
