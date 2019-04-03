from functools import partial

import keras
from keras import backend as K
from keras.layers import Layer
from operator import truediv
from keras.callbacks import Callback
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

from model import *
from dataset import *


def true_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))


def possible_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_true, 0, 1)))


def predicted_positives(y_true, y_pred):
    return K.sum(K.round(K.clip(y_pred, 0, 1)))


class F1Callback(Callback):
    def __init__(self):
        super(F1Callback, self).__init__()
        self.f1s = []
        self.val_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        eps = np.finfo(np.float32).eps
        recall = logs["val_true_positives"] / (logs["val_possible_positives"] + eps)
        precision = logs["val_true_positives"] / (logs["val_predicted_positives"] + eps)
        self.val_f1 = 2 * precision * recall / (precision + recall + eps)
        print("f1_val (from log) =", self.val_f1)
        self.f1s.append(self.val_f1)


# https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/
class Histories(keras.callbacks.Callback):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.aucs = []
        self.losses = []
        self.auc = 0.0
        self.dataset = dataset

    def on_train_begin(self, logs=None):
        if logs is None:
            logs = dict()
        self.losses = []
        self.aucs = []

    def on_train_end(self, logs=None):
        if logs is None:
            logs = {}
        return

    def on_epoch_begin(self, epoch, logs=None):
        if logs is None:
            logs = {}
        return

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))
        validate_steps = int(np.floor(float(self.dataset.validate_num) / float(BATCH_SIZE)))
        y_pred = self.model.predict_generator(generator=train_utils.next_simple_dataset(self.dataset, BATCH_SIZE, DataType.validate),
                                              steps=validate_steps)
        auc = roc_auc_score(self.model.validation_data[1], y_pred)
        is_best = False
        if auc > self.auc:
            self.auc = auc
            is_best = True
        print(f"val_auc:{auc} {'best' if is_best else ''}")
        self.aucs.append(self.auc)
        return

    def on_batch_begin(self, batch, logs=None):
        if logs is None:
            logs = {}
        return

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        return


# https://github.com/keras-team/keras/issues/6050
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    value, update_op = tf.metrics.auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


# https://stackoverflow.com/questions/41032551/how-to-compute-receiving-operating-characteristic-roc-and-auc-in-keras
def auc(y_true, y_pred):
    value = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    # print(f"y_true:{tf.shape(y_true)} y_pred:{tf.shape(y_pred)} auc:{auc}")
    return value
