import numpy as np
from keras import backend
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import pickle
from configparser import ConfigParser


def main():
    print(backend.tensorflow_backend._get_available_gpus())
    
    config = ConfigParser()
    config.read('config.ini')

    model_folder = config.get('DATA FOLDER', 'model')
    preprocessed_folder = config.get('DATA FOLDER', 'preprocessed')

    model = load_model(f'{model_folder}/deepcnn.hdf5')

    x_data = pickle.load(open(f"{preprocessed_folder}/x_test.p", "rb"))
    y_data = np.array(pickle.load(open(f"{preprocessed_folder}/y_test.p", "rb")))
    x_data_padded = pad_sequences(x_data, maxlen=512, dtype='float', padding='post')

    accuracy = model.evaluate(x_data_padded, y_data, batch_size=32)[1]
    print(f'accuracy on the test set is {100*accuracy}%')

if __name__ == "__main__":
    main()
