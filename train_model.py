from keras import backend, regularizers
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Conv1D, Dense, Dropout, GlobalAveragePooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from configparser import ConfigParser


def main():
    # see if gpu is available
    print(backend.tensorflow_backend._get_available_gpus())

    config = ConfigParser()
    config.read('config.ini')

    model_folder = config.get('DATA FOLDER', 'model')
    preprocessed_folder = config.get('DATA FOLDER', 'preprocessed')

    # load data
    embedding_dims = 300
    x_data = pickle.load(open(f'{preprocessed_folder}/x_train.p', 'rb'))
    y_data = np.array(pickle.load(open(f'{preprocessed_folder}/y_train.p',
        'rb')))
    x_data_padded = pad_sequences(x_data, maxlen=512, dtype='float',
        padding='post')
    x_train, x_val, y_train, y_val, idx_train, idx_val = train_test_split(
        x_data_padded, y_data, np.arange(len(x_data)), test_size=0.1,
        random_state=42)

    # model parameters
    dropout = 0.6
    l2_reg = 1e-4

    model = Sequential()
    model.add(Conv1D(filters=256, kernel_size=3,
        activation='relu', input_shape=(512, embedding_dims),
        padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', 
        padding='same', dilation_rate=2,
        kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', 
        padding='same', dilation_rate=4,
        kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(Dropout(dropout))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu', 
        padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(dropout))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=adam,
        metrics=['accuracy'])
    model.summary()

    # training parameters
    batch_size = 128
    epochs = 20
    model_checkpoint = ModelCheckpoint(f'{model_folder}/deepcnn.hdf5',
        monitor='val_acc', save_best_only=True)
    callbacks_list = [model_checkpoint]

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              callbacks=callbacks_list, validation_data=(x_val, y_val))

    print(f'wrote model {model_folder}/deepcnn.hdf5')

if __name__ == "__main__":
    main()
