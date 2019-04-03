#!/usr/bin/env python

import mpl_toolkits # import before pathlib
import sys
import pathlib
import gc
from typing import Optional

from sklearn.neighbors import NearestNeighbors
from tensorflow import set_random_seed

# sys.path.append(pathlib.Path(__file__).parent)
from train_utils import *
from model import *
from dataset import *
from predict_test import test

np.random.seed(RANDOM_NUM)
set_random_seed(RANDOM_NUM)

OUTPUT_FILE = 'test_dataset_prediction.txt'

# BASE_MODEL = 'vgg19'
# BASE_MODEL = 'incepstionresnetv2'
# BASE_MODEL = 'resnet50'
# BASE_MODEL = 'resnet152'
# BASE_MODEL = 'nasnet'
BASE_MODEL = 'ensemble'
if BASE_MODEL == 'resnet50':
    create_model = create_model_resnet50_plain
elif BASE_MODEL == 'resnet152':
    create_model = create_model_resnet152_plain
elif BASE_MODEL == 'incepstionresnetv2':
    create_model = create_model_inceptionresnetv2_plain
elif BASE_MODEL == 'mobilenet':
    create_model = create_model_mobilenet
elif BASE_MODEL == 'nasnet':
    create_model = create_model_nasnet
elif BASE_MODEL == 'ensemble':
    create_model = create_model_ensemble
else:
    raise Exception("unimplemented model")


def main():
    dataset = load_raw_data()
    weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
    dataset.weight_param_path = weight_param_path
    model = create_model(dataset=dataset, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM))
    model = build_model(model, weight_param_path)
    # model = create_martine_model()
    for i in range(0, 1):
        print(f"num:{i}. start train")
        train_model(model, dataset, weight_param_path)
    # model.save(weight_param_path)
    test(model, dataset)


if __name__ == "__main__":
    main()
