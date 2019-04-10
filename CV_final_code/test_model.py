import numpy
import keras
from keras.models import load_model
from data_gen import DataGenerator
import functools
from cat_parse import get_vector_from_cats

top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)
top3_acc.__name__ = 'top3_acc'
model = load_model('./VGG19_weights/aug_bce_rms.h5', custom_objects={'top3_acc':top3_acc})
model.summary()
print(model.get_layer(index=3))



