import os
import pickle
import sys
import pathlib
from pathlib import Path
import csv
from enum import Enum
from typing import Union
import uuid
import copy
import threading

import pandas as pd
import numpy as np
from numpy.linalg import inv as mat_inv
from PIL import Image
from PIL import ImageOps
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import pairwise_distances
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
from skmultilearn import model_selection
import keras
from keras import backend as K
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

import train_utils


RANDOM_NUM = 77777
IMAGE_BASE_DIR = '../cancer_detection_dataset'
TRAIN_DIRS = [f"{IMAGE_BASE_DIR}/train", ]
TRAIN_ANSWER_FILES = ['train_labels.csv', ]
TEST_DIR = IMAGE_BASE_DIR + '/test'
# dataset ratio
TRAIN_RATIO = 0.9
VALIDATE_RATIO = 0.05
TEST_RATIO = 0.05

ORG_IMAGE_SIZE = 96
ORG_ROI_SIZE = 32
ROTATED_ROI_SIZE = np.ceil(ORG_ROI_SIZE * (2 ** 0.5))
AUG_SCALE = 0.2
# 32:46
ROI_IMAGE_SIZE = int(np.ceil(ROTATED_ROI_SIZE * (1 + AUG_SCALE)))
IMAGE_SIZE = 256
# IMAGE_SIZE = 76
print(f"train image size:{IMAGE_SIZE}")
IMAGE_DIM = 3
BATCH_SIZE = 5
EPOCHS = 200


class DataType(Enum):
    train = 1
    validate = 2
    test = 3


class Dataset(Callback):
    def __init__(self, data_list: np.array):
        super(Dataset, self).__init__()
        self.data_list = data_list
        np.random.shuffle(self.data_list)
        self.sample_num = len(self.data_list)
        self.train_num = int(len(self.data_list) * TRAIN_RATIO)
        self.validate_num = int(len(self.data_list) * VALIDATE_RATIO)
        self.test_num = int(len(self.data_list) * TEST_RATIO * 0.5)
        self.train_counter = 0
        self.validate_counter = 0
        self.test_counter = 0
        self.test_index_list = list(range(self.train_num + self.validate_num, self.sample_num))
        self.model = None
        self.best_auc = 0
        self.weight_param_path = None

    def on_train_begin(self, logs=None):
        self.train_counter = 0
        self.validate_counter = 0
        self.test_counter = 0
        np.random.shuffle(self.test_index_list)

    def on_epoch_end(self, epoch, logs=None):
        test_steps = int(np.floor(float(self.test_num) / float(BATCH_SIZE)))
        y_pred = self.model.predict_generator(generator=train_utils.next_simple_dataset(self, BATCH_SIZE, DataType.test),
                                              steps=test_steps)
        # y_pred = np.rint(y_pred.flatten())
        y_pred = np.ravel(y_pred)
        y_true = [int(self.data_list[idx].answer) for idx in self.test_index_list]
        y_true = np.array(y_true[:len(y_pred)])
        print(f"test_steps:{test_steps} y_pred:{y_pred.shape}:{y_pred} y_true:{y_true.shape}")
        auc = metrics.roc_auc_score(y_true, y_pred)
        # fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        print(f"test_roc_auc_score:{auc}")
        if auc > self.best_auc and self.weight_param_path:
            self.best_auc = auc
            self.model.save(self.weight_param_path)
            print(f"model saved. {self.weight_param_path}")

        # plt.figure(1)
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(fpr, tpr, label='area = {:.3f}'.format(auc))
        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.title('ROC curve')
        # plt.legend(loc='best')
        # plt.show()

        # initialize counter
        self.train_counter = 0
        self.validate_counter = 0
        self.test_counter = 0
        np.random.shuffle(self.test_index_list)

    def increment_train(self):
        self.train_counter = (self.train_counter + 1) % self.train_num

    def increment_validate(self):
        self.validate_counter = (self.validate_counter + 1) % self.validate_num

    def increment_test(self):
        if self.test_counter + 1 == self.test_num:
            raise StopIteration
        self.test_counter = (self.test_counter + 1) % self.test_num

    def next_train_data(self):
        # self.increment_train()
        index = np.random.randint(self.test_num)
        data_unit = self.data_list[index]
        return data_unit, index

    def next_validate_data(self):
        # self.increment_validate()
        index = self.train_num + np.random.randint(self.validate_num)
        data_unit = self.data_list[index]
        return data_unit, index

    def next_test_data(self):
        index = self.test_index_list[self.test_counter]
        self.increment_test()
        data_unit = self.data_list[index]
        return data_unit, index


class TestDataset:
    def __init__(self, data_list: list):
        self.data_list = data_list


class DataUnit:
    def __init__(self, filename: str, answer: Union[str, None], source_dir: str):
        self.filename = filename
        self.answer = answer
        self.source_dir = source_dir


def load_raw_data():
    data_list = []
    answers = []
    for idx, train_answer_file in enumerate(TRAIN_ANSWER_FILES):
        source_dir = TRAIN_DIRS[idx]
        with open(train_answer_file, 'r') as f:
            csv_reader = csv.reader(f)
            # skip header
            next(csv_reader)
            # read lines
            for row in csv_reader:
                if len(row) != 2:
                    continue
                filename = row[0]
                answer = row[1].strip()
                answers.append(float(answer))
                # print(f"filename:{filename} answer:{answer}")
                data_unit = DataUnit(filename, answer, source_dir)
                data_list.append(data_unit)

    dataset = Dataset(np.array(data_list))
    return dataset


def load_test_data():
    data_list = []
    for filename in os.listdir(TEST_DIR):
        data_unit = DataUnit(filename, None, TEST_DIR)
        data_list.append(data_unit)
    return TestDataset(data_list)


def create_xy(dataset: Dataset, datatype: DataType):
    if datatype == DataType.train:
        data_unit, index = dataset.next_train_data()
    elif datatype == DataType.validate:
        data_unit, index = dataset.next_validate_data()
    elif datatype == DataType.test:
        data_unit, index = dataset.next_test_data()
    else:
        raise RuntimeError(f"invalid data type. type={str(datatype)}")
    # print(f"create_xy data_unit:{data_unit.answer}")
    x = generate_input(data_unit)
    # x = normalize_image(x, data_unit)
    # y = dataset.classes.tolist().index(data_unit.answer)
    y = data_unit.answer
    # print(f"create_training_sample x:{x.shape} y:{y.shape}")
    return x, y, data_unit, index


def generate_input(data_unit: DataUnit, add_extension=True):
    if add_extension:
        file_path = Path(data_unit.source_dir, data_unit.filename + '.tif')
    else:
        file_path = Path(data_unit.source_dir, data_unit.filename)
    img = Image.open(str(file_path))
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img = np.array(img)
    return img


def create_unit_dataset(data_unit: DataUnit):
    x = generate_input(data_unit, add_extension=False)
    x = x.astype("float32")
    # corner = (x.shape[0] - IMAGE_SIZE) // 2
    # x = x[corner:(x.shape[0] - corner), corner:(x.shape[0] - corner)]
    return normalize(x.reshape(1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM))


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


# Read an image for validation, i.e. without data augmentation.
def normalize(x):
    # x -= np.mean(x, keepdims=True)
    # x /= np.std(x, keepdims=True) + K.epsilon()
    # x = (x - x.min()) / (x.max() - x.min() + K.epsilon())
    return x.astype(np.float32) / 255.0
