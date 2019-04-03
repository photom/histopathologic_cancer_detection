import string
import random
import sys
import pickle
import pathlib
import time

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import imgaug as ia
from imgaug import augmenters as iaa

# sys.path.append(pathlib.Path(__file__).parent)
from dataset import *
import metrics


def create_seq():
    SEQ = iaa.Sequential([
        iaa.OneOf([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Flipud(0.5),  # vertically flips
            iaa.Crop(percent=(0, 0.2)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5,
                          iaa.GaussianBlur(sigma=(0, 0.5))
                          ),
            # Strengthen or weaken the contrast in each image.
            iaa.ContrastNormalization((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=True),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (1.0 - AUG_SCALE, 1.0 + AUG_SCALE), "y": (1.0 - AUG_SCALE, 1.0 + AUG_SCALE)},
                # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-180, 180),
                shear=(-8, 8),
            )
        ])], random_order=True)
    return SEQ


def create_callbacks(dataset, name_weights, patience_lr=10, patience_es=150):
    mcp_save = ModelCheckpoint('model/validate.weights.best.hdf5', save_best_only=True, monitor='val_loss')
    # history = metrics.Histories(dataset)
    # mcp_save = AllModelCheckpoint(name_weights)
    # reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
    # early_stopping = EarlyStopping(monitor='val_loss', patience=patience_es, verbose=1, mode='auto')
    # return [early_stopping, mcp_save, reduce_lr_loss]
    # return [f1metrics, early_stopping, mcp_save]
    return [dataset, mcp_save]


def load_dataset(filename):
    with open(filename, 'rb') as f:
        # The protocol version used is detected automatically, so we do not
        # have to specify it.
        train_dataset = pickle.load(f)
    return train_dataset


def next_simple_dataset(dataset, batch_size: int, datatype):
    """ Obtain a batch of training data
    """
    while True:
        x_batch = []
        y_batch = []
        for i in range(batch_size):
            try:
                x, y, data_unit, index = create_xy(dataset, datatype)
                # x = normalize(x)
                x_batch.append(x)
                y_batch.append(y)
            except StopIteration:
                break
        x_batch, y_batch = np.array(x_batch), np.array(y_batch)
        if datatype != DataType.test:
            x_batch = SEQ_CVXTZ.augment_images(x_batch).astype("float32")
        x_batch = np.array([normalize(x) for x in x_batch])
        # org_shape = x_batch.shape
        # org_width = x_batch.shape[1]
        # corner = int((org_width - ROI_IMAGE_SIZE) // 2)
        # print(f"0: org_shape:{org_shape} x_batch:{x_batch.shape} corner:{corner}")
        # x_batch = x_batch[:, corner:(org_width - corner), corner:(org_width - corner), :]
        # resized_x_batch = []
        # for x in x_batch:
        #     img = Image.fromarray(np.uint8(x))
        #     img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        #     resized_x_batch.append(normalize(np.array(img)))
        # print(f"1: org_shape:{org_shape} corner:{corner} x_batch:{x_batch.shape}")
        # yield np.array(resized_x_batch), y_batch
        yield np.array(x_batch), y_batch


def train_model(model: Model, dataset, model_filename: str,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS, ):
    callbacks = create_callbacks(dataset, model_filename)
    dataset.model = model
    answers = [data_unit.answer for data_unit in dataset.data_list]
    sample_num = len(answers)
    # sample_num = len(answers)
    train_num = int(sample_num * TRAIN_RATIO)
    validate_num = int(sample_num * VALIDATE_RATIO)
    steps_per_epoch = train_num // batch_size
    # steps_per_epoch = 50
    validation_steps = validate_num // batch_size
    print(f"train_num:{train_num} validate_num:{validate_num} steps_per_epoch:{steps_per_epoch} validateion_steps:{validation_steps}")
    model.fit_generator(generator=next_simple_dataset(dataset, batch_size, DataType.train),
                        epochs=epochs,
                        validation_data=next_simple_dataset(dataset, batch_size, DataType.validate),
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps,
                        callbacks=callbacks, verbose=1)


def create_sequential_cvxtz():
    # https://www.kaggle.com/CVxTz/cnn-starter-nasnet-mobile-0-9709-lb
    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.2),  # vertically flip 20% of all images
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},  # translate by -20 to +20 percent (per axis)
                rotate=(-10, 10),  # rotate by -45 to +45 degrees
                shear=(-5, 5),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                       [
                           sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                           # convert images into their superpixel representation
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 1.0)),  # blur images with a sigma between 0 and 3.0
                               iaa.AverageBlur(k=(3, 5)),
                               # blur image using local means with kernel sizes between 2 and 7
                               iaa.MedianBlur(k=(3, 5)),
                               # blur image using local medians with kernel sizes between 2 and 7
                           ]),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)),  # sharpen images
                           iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                           # search either for all edges or for directed edges,
                           # blend the result with the original image using a blobby mask
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.5, 1.0)),
                               iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.5),
                           # add gaussian noise to images
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                           ]),
                           iaa.Invert(0.01, per_channel=True),  # invert color channels
                           iaa.Add((-2, 2), per_channel=0.5),
                           # change brightness of images (by -10 to 10 of original value)
                           iaa.AddToHueAndSaturation((-1, 1)),  # change hue and saturation
                           # either change the brightness of the whole image (sometimes
                           # per channel) or change the brightness of subareas
                           iaa.OneOf([
                               iaa.Multiply((0.9, 1.1), per_channel=0.5),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-1, 0),
                                   first=iaa.Multiply((0.9, 1.1), per_channel=True),
                                   second=iaa.ContrastNormalization((0.9, 1.1))
                               )
                           ]),
                           sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                           # move pixels locally around (with random strengths)
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           # sometimes move parts of the image around
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )
    return seq


SEQ_CVXTZ = create_sequential_cvxtz()
