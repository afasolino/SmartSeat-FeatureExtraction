# model_def.py
import os
import argparse
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, GlobalAveragePooling2D, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import logging
import scipy.ndimage
import sys
import seaborn as sns
import io

# Custom F1-Score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Modified SqueezeNet Model
def build_modified_squeezenet(input_shape=(64, 158, 1), num_classes=20, l_reg=0.005):
    inputs = Input(shape=input_shape)
    kernel_regularizer=l2(l_reg)
    # Initial Conv Layer
    x = Conv2D(96, (7, 7), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    # Replace Fire modules with equivalent standard layers
    x = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(64, (1, 1), activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (1, 1), activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, (1, 1), activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, (1, 1), activation='relu', kernel_regularizer=kernel_regularizer)(x)
    x = BatchNormalization()(x)

    # Final layers
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model