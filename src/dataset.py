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
from os.path import isfile
from math import sqrt
import asyncio

import pandas as pd
import numpy as np
from numpy.linalg import inv as mat_inv
from PIL import Image
from PIL import ImageOps
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import pairwise_distances
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
from sklearn import model_selection
import keras
from keras import backend as K
from keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from imagehash import phash
from tqdm import tqdm

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
IMAGE_SIZE = int(256)
# IMAGE_SIZE = 76
print(f"train image size:{IMAGE_SIZE}")
IMAGE_DIM = 3
BATCH_SIZE = 15
EPOCHS = 30


class DataType(Enum):
    train = 1
    validate = 2
    test = 3


class Dataset(Callback):
    def __init__(self, total_data_list: np.array):
        super(Dataset, self).__init__()
        np.random.shuffle(total_data_list)
        self.sample_num = len(total_data_list)
        self.train_num = int(len(total_data_list) * TRAIN_RATIO)
        self.validate_num = int(len(total_data_list) * VALIDATE_RATIO)
        self.test_num = int(len(total_data_list) * TEST_RATIO)
        self.train_counter = 0
        self.validate_counter = 0
        self.test_counter = 0
        sss = model_selection.StratifiedShuffleSplit(n_splits=2, test_size=self.test_num,
                                                     random_state=RANDOM_NUM)
        total_answers = [data_unit.answer for data_unit in total_data_list]
        self.train_validate_index_list, self.test_index_list = next(sss.split(total_data_list, total_answers))
        self.test_data_list = [total_data_list[idx] for idx in self.test_index_list]
        self.data_list = [total_data_list[idx] for idx in self.train_validate_index_list]
        self.answers = [data_unit.answer for data_unit in self.data_list]
        self.sss = model_selection.StratifiedShuffleSplit(n_splits=101, test_size=self.validate_num,
                                                          random_state=RANDOM_NUM)
        self.train_index_list, self.validate_index_list = next(self.sss.split(self.data_list, self.answers))
        self.model = None
        self.best_auc = 0
        self.weight_param_path = None
        self.image2hash = None
        self.hash2images = None
        self.hash2y = None
        for _ in range(31):
            self.train_index_list, self.validate_index_list = next(self.sss.split(self.data_list, self.answers))

    def on_epoch_begin(self, epoch, logs=None):
        self.train_counter = 0
        self.validate_counter = 0
        self.test_counter = 0
        self.train_index_list, self.validate_index_list = next(self.sss.split(self.data_list, self.answers))
        print(f"epoch begin. train_index_list_size:{len(self.train_index_list)} train_num:{self.train_num} validate_index_list_size:{len(self.validate_index_list)} validate_num:{self.validate_num}")

    def on_epoch_end(self, epoch, logs=None):
        test_steps = int(np.floor(float(self.test_num) / float(BATCH_SIZE)))
        y_pred = self.model.predict_generator(
            generator=train_utils.next_simple_dataset(self, BATCH_SIZE, DataType.test),
            steps=test_steps)
        # y_pred = np.rint(y_pred.flatten())
        y_pred = np.ravel(y_pred)
        y_true = [int(data_unit.answer) for data_unit in self.test_data_list]
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

    def increment_train(self):
        self.train_counter = (self.train_counter + 1) % len(self.train_index_list)

    def increment_validate(self):
        self.validate_counter = (self.validate_counter + 1) % len(self.validate_index_list)

    def increment_test(self):
        if self.test_counter + 1 == self.test_num:
            raise StopIteration
        self.test_counter = (self.test_counter + 1) % self.test_num

    def next_train_data(self):
        index = self.train_index_list[self.train_counter]
        data_unit = self.data_list[index]
        self.increment_train()
        return data_unit, index

    def next_validate_data(self):
        index = self.validate_index_list[self.validate_counter]
        data_unit = self.data_list[index]
        self.increment_validate()
        return data_unit, index

    def next_test_data(self):
        data_unit = self.test_data_list[self.test_counter]
        self.increment_test()
        return data_unit, self.test_counter


class TestDataset:
    def __init__(self, data_list: list):
        self.data_list = data_list


class DataUnit:
    def __init__(self, filename: str, answer: Union[str, None], source_dir: str):
        self.filename = filename
        if filename[-4: -1] == '.tif':
            self.filename += '.tif'
        self.answer = answer
        self.source_dir = source_dir


def load_raw_data():
    if isfile('image2hash.pickle') and isfile('hash2images.pickle'):
        with open('image2hash.pickle', 'rb') as f:
            image2hash = pickle.load(f)
        with open('hash2images.pickle', 'rb') as f:
            hash2images = pickle.load(f)
    else:
        image2hash, hash2images = create_phash()
        with open('image2hash.pickle', 'wb') as f:
            pickle.dump(image2hash, f)
        with open('hash2images.pickle', 'wb') as f:
            pickle.dump(hash2images, f)

    data_list = []
    answers = []
    hash2y = {}
    for idx, train_answer_file in enumerate(TRAIN_ANSWER_FILES):
        source_dir = TRAIN_DIRS[idx]
        with open(train_answer_file, 'r') as f:
            csv_reader = csv.reader(f)
            # skip header
            next(csv_reader)
            # read lines
            print(f"reading {train_answer_file} ...")
            for row in tqdm(csv_reader):
                if len(row) != 2:
                    continue
                filename = row[0]
                answer = row[1].strip()
                tif = filename + '.tif'
                h = image2hash[tif]
                if tif in hash2y and hash2y[tif] != answer:
                    print(f"dup hashes with different result. filename:{filename} dup_imgs:{hash2images[h]}")
                hash2y[h] = answer
                if h not in hash2images:
                    hash2images[h] = set()
                    hash2images[h].add(tif)
                answers.append(float(answer))
                # print(f"filename:{filename} answer:{answer}")
                data_unit = DataUnit(tif, answer, source_dir)
                data_list.append(data_unit)

    dataset = Dataset(np.array(data_list))
    dataset.image2hash = image2hash
    dataset.hash2images = hash2images
    dataset.hash2y = hash2y
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
    h = dataset.image2hash[data_unit.filename]
    hy = dataset.hash2y[h]
    y = data_unit.answer
    if hy != y:
        print(f"miss match y_true. hy:{hy} y:{y} filename:{data_unit.filename} dup_imgs:{dataset.hash2images[h]}")
        if data_unit.filename == '4e096041cae856c608360cb70bca8dbcd9006f1f.tif' or \
                data_unit.filename == 'e0d7d76ac1bccca03af9a53ca9d25aaf97e676d7.tif' or \
                data_unit.filename == '62e2769adf92235fe2bf439865fd7ebc70922fa3.tif' or \
                data_unit.filename == '5de41692e28d443f08740e83062a1339f62a98ab.tif':
            dataset.hash2y[h] = 0
            hy = 0
    return x, hy, data_unit, index


def open_image(data_unit: DataUnit):
    file_path = Path(data_unit.source_dir, data_unit.filename)
    img = Image.open(str(file_path))
    return img


def generate_input(data_unit: DataUnit):
    img = open_image(data_unit)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img = np.array(img)
    return img


def create_unit_dataset(data_unit: DataUnit):
    x = generate_input(data_unit)
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


def expand_path(filename):
    if isfile('./train/' + filename):
        return './train/' + filename
    if isfile('./test/' + filename):
        return './test/' + filename
    return filename


# Two phash values are considered duplicate if, for all associated image pairs:
# 1) They have the same mode and size;
# 2) After normalizing the pixel to zero mean and variance 1.0, the mean square error does not exceed 0.1
def match(h1, h2, hash2images):
    for p1 in hash2images[h1]:
        for p2 in hash2images[h2]:
            i1 = Image.open(expand_path(p1))
            i2 = Image.open(expand_path(p2))
            if i1.mode != i2.mode or i1.size != i2.size:
                return False
            a1 = np.array(i1)
            a1 = a1 - a1.mean()
            s = sqrt((a1 ** 2).mean())
            if s:
                a1 = a1 / s
            a2 = np.array(i2)
            a2 = a2 - a2.mean()
            s = sqrt((a2 ** 2).mean())
            if s:
                a2 = a2 / s
            a = ((a1 - a2) ** 2).mean()
            if a > 0.1:
                return False
    return True


def create_phash():
    # Compute phash for each image in the training and test set.
    image2hash = {}

    async def gen_phash(filename):
        img = Image.open(expand_path(filename))
        h = phash(img)
        image2hash[filename] = h
    loop = asyncio.get_event_loop()

    async def gen_phashes():
        for d in (TRAIN_DIRS + [TEST_DIR]):
            print(f"creating phash dir:{d} ...")
            tasks = []
            for filename in tqdm(os.listdir(d)):
                task = asyncio.ensure_future(gen_phash(filename))
                tasks.append(task)
            await asyncio.gather(*tasks, return_exceptions=True)
    loop.run_until_complete(gen_phashes())
    print(f"image2hash:{len(image2hash)}")

    # Find all images associated with a given phash value.
    print("creating hash2img ...")
    hash2images = {}
    for p, h in tqdm(image2hash.items()):
        if h not in hash2images:
            hash2images[h] = set()
        if p not in hash2images[h]:
            hash2images[h].add(p)

    # Find all distinct phash values
    hashes = list(hash2images.keys())

    # If the images are close enough, associate the two phash values (this is the slow part: n^2 algorithm)
    print("creating hash2hash ...")
    hash2hash = {}

    async def gen_hash2hash(idx, h1):
        h2h = {}
        for h2 in hashes[:idx]:
            if h1 - h2 <= 6 and match(h1, h2, hash2images):
                s1 = str(h1)
                s2 = str(h2)
                if s1 < s2:
                    s1, s2 = s2, s1
                h2h[s1] = s2
        return h2h

    async def gen_hash2hashes():
        cors = []
        for i, h1 in enumerate(tqdm(hashes)):
            cors.append(gen_hash2hash(i, h1))
        return await asyncio.gather(*cors)
    loop = asyncio.get_event_loop()
    h2h_list = loop.run_until_complete(gen_hash2hashes())
    print(f"h2h_list:{len(h2h_list)}")
    print("merging hash2hash ...")
    for h2h in tqdm(h2h_list):
        for s1, s2 in h2h.items():
            if s1 not in hash2hash:
                hash2hash[s1] = s2
            elif s2 > hash2hash[s1]:
                hash2hash[s1] = s2
    print(f"hash2hash:{len(hash2hash)}")
    # Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    print("reformatting image2hash ...")
    for p, h in tqdm(image2hash.items()):
        h = str(h)
        if h in hash2hash:
            h = hash2hash[h]
        image2hash[p] = h

    return image2hash, hash2images
