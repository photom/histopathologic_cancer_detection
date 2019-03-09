import mpl_toolkits  # import before pathlib
import sys
from pathlib import Path

from tensorflow import set_random_seed
from sklearn.neighbors import NearestNeighbors

# sys.path.append(Path(__file__).parent)
from train_utils import *
from model import *
from dataset import *

np.random.seed(RANDOM_NUM)
set_random_seed(RANDOM_NUM)

OUTPUT_FILE = 'test_dataset_prediction.txt'
BASE_MODEL = 'resnet50'
# BASE_MODEL = 'vgg11'
# BASE_MODEL = 'incepstionresnetv2'
# BASE_MODEL = 'nasnet'
# BASE_MODEL = 'adams'
# BASE_MODEL = 'michel'
# BASE_MODEL = 'giim'
# BASE_MODEL = 'siamese_resnet'
# BASE_MODEL = 'triplet_loss'
if BASE_MODEL == 'resnet50':
    create_model = create_model_resnet50_plain
elif BASE_MODEL == 'incepstionresnetv2':
    create_model = create_model_inceptionresnetv2_plain
elif BASE_MODEL == 'nasnet':
    create_model = create_model_nasnet
else:
    raise Exception("unimplemented model")


def test(test_model=None, dataset=None):
    if not dataset:
        dataset = load_raw_data()
    # weight_param_path = f"model/{BASE_MODEL}.weights.best.hdf5"
    weight_param_path = "model/resnet50.weights.best.hdf5.bak"
    if not test_model:
        test_model = create_model(dataset, input_shape=(IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM), datatype=DataType.test)
        test_model.load_weights(weight_param_path)
    test_dataset = load_test_data()
    content = "id,label\n"
    splitted = np.array_split(np.array(test_dataset.data_list), len(test_dataset.data_list) // BATCH_SIZE)
    for idx, data_list in enumerate(splitted):
        x = []
        for data_unit in data_list:
            x.append(create_unit_dataset(data_unit))
        x = np.array(x)
        y_pred = test_model.predict(x.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DIM))
        y_pred = y_pred.flatten()
        print(f"idx:{idx} y_pred:{y_pred}")
        for data_unit, y in zip(data_list, y_pred):
            content += f"{data_unit.filename[:40]},{y}\n"

    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    test()
