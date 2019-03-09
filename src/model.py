import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Dropout, TimeDistributed, \
    Conv3D, Conv2D, Conv1D, Flatten, MaxPooling1D, MaxPooling3D, MaxPooling2D, \
    GlobalAveragePooling2D, Layer, GlobalMaxPooling2D, AveragePooling2D
from keras.layers import GRU, Bidirectional, BatchNormalization
from keras.layers import Input, ELU, Lambda
from keras.layers import Reshape
from keras.optimizers import Adam, Nadam
from keras import backend as keras_backend
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg19 import VGG19
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras_contrib.applications import resnet
from keras_contrib.applications.nasnet import NASNetMobile
from keras_contrib.applications.resnet import ResNet152, ResNet
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras_contrib.layers import advanced_activations
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, \
    Concatenate, ReLU, LeakyReLU
import tensorflow as tf
from keras import regularizers
from keras.optimizers import Adam
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    Lambda, MaxPooling2D, Reshape
from keras.models import Model

from dataset import *
from train_utils import *
from metrics import MacroF1Score


def elu(x, alpha=0.05):
    return K.elu(x, alpha)


def create_model_resnet50_plain(dataset: Dataset, input_shape, dropout=0.3, datatype: DataType = DataType.train):
    """
    loss: 0.2251 - acc: 0.9082 - val_loss: 0.3021 - val_acc: 0.8824 test auc: 0.5168897947617596
    loss: 0.2697 - acc: 0.8866 - val_loss: 0.3675 - val_acc: 0.8455 roc_auc_score:0.9416684622566974
    loss: 0.1372 - acc: 0.9447 - val_loss: 0.3733 - val_acc: 0.8818 roc_auc_score:0.9762951119646599 0.9429
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    base_input = Input(shape=input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=base_input, pooling=None)
    # base_model = ResNet50(weights=None, include_top=False, input_tensor=base_input, pooling=None)
    # x = GlobalAveragePooling2D()(base_model.layers[-1].output)
    out1 = GlobalMaxPooling2D()(base_model.layers[-1].output)
    out2 = GlobalAveragePooling2D()(base_model.layers[-1].output)
    out3 = Flatten()(base_model.layers[-1].output)
    out = Concatenate(axis=-1)([out1, out2, out3])
    if datatype != DataType.test:
        out = Dropout(dropout)(out)
    x = Dense(1, activation="sigmoid")(out)
    model = Model(inputs=[base_model.input], outputs=[x])
    model.summary()
    return model


def create_model_resnet152_plain(dataset: Dataset, input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    model = ResNet(input_shape, activation='sigmoid', classes=1,
                   repetitions=[3, 8, 36, 3], include_top=True)
    model.summary()
    return model


def create_model_inceptionresnetv2_plain(dataset: Dataset, input_shape, dropout=0.5,
                                         datatype: DataType = DataType.train):
    """
    loss: 0.1378 - acc: 0.9448 - val_loss: 0.3922 - val_acc: 0.8667 test_auc:0.964241981312196
    loss: 0.3330 - acc: 0.8532 - val_loss: 0.4429 - val_acc: 0.8358 test_auc:0.9377751764453554

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    base_input = Input(shape=input_shape)
    base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_tensor=base_input, pooling=None)
    x = GlobalAveragePooling2D()(base_model.layers[-1].output)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[base_model.input], outputs=[x])
    model.summary()
    return model


def create_model_nasnet(dataset: Dataset, input_shape, dropout=0.5,
                                         datatype: DataType = DataType.train):
    # loss: 0.3772 - acc: 0.9286 - val_loss: 0.6968 - val_acc: 0.8291 roc_auc_score:0.9395888617041388 0.8966
    # loss: 0.4615 - acc: 0.9066 - val_loss: 0.7080 - val_acc: 0.8309 roc_auc_score:0.934900486
    # loss: 0.4558 - acc: 0.9053 - val_loss: 0.6908 - val_acc: 0.8309 roc_auc_score:0.9408894702510078 0.8956
    inputs = Input(input_shape)
    base_model = NASNetMobile(include_top=False, input_shape=input_shape)  # , weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(dropout)(out)
    out = Dense(1, activation="sigmoid", name="3_")(out)
    model = Model(inputs, out)
    # model.compile(optimizer=Adam(0.0001), loss=binary_crossentropy, metrics=['acc'])
    model.summary()
    return model


def create_model_mobilenet(dataset: Dataset, input_shape, dropout=0.5, datatype: DataType = DataType.train):
    """
    Function creating the model's graph in Keras.
    loss: 0.8077 - f1_score: 0.4574 - val_loss: 0.8119 - val_f1_score: 0.4471
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)
    Returns:
    model -- Keras model instance
    """
    base_input = Input(shape=input_shape)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=base_input, pooling=None)
    x = GlobalAveragePooling2D()(base_model.layers[-1].output)
    if datatype != DataType.test:
        x = Dropout(dropout)(x)
    x = Dense(dataset.class_num, activation='softmax')(x)
    model = Model(inputs=[base_model.input], outputs=[x])
    model.summary()
    return model


def build_model(model: Model, model_filename: str = None, learning_rate=0.00005):
    if model_filename and os.path.exists(model_filename):
        print(f"load weights: file={model_filename}")
        model.load_weights(model_filename)

    opt = Nadam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
    model.compile(
        # optimizer=keras.optimizers.Adadelta(),
        optimizer=opt,
        # loss=identity_loss,
        # loss=keras.losses.mean_absolute_error,
        loss=keras.losses.binary_crossentropy,
        # loss=keras.losses.categorical_crossentropy,
        # loss=max_average_precision,
        metrics=['acc'], )
    # )

    return model
