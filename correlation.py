import numpy as np
from keras import backend
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from itertools import combinations
import pickle

from explanations import create_explanations


def explanation_correlation(expl1, expl2, indices):
    number_of_explanations = min(expl1.shape[0], expl2.shape[0])
    correlation_sum = 0
    for idx in range(number_of_explanations):
        len_input = len(x_data[indices[idx]])
        correlation_sum += np.corrcoef(expl1[idx][:len_input].flatten(),
            expl2[idx][:len_input].flatten())[0][1]
    return correlation_sum / number_of_explanations

def main():
    print(backend.tensorflow_backend._get_available_gpus())

    np.random.seed(42)
    thousand_indices = np.random.choice(25000, 1000, replace=False)

    x_data = pickle.load(open("/mnt/hdd/datasets/imdb_glove_pickle/x_test.p",
        "rb"))
    y_test = np.array(pickle.load(open("/mnt/hdd/datasets/imdb_glove_pickle/"+\
        "y_test.p", "rb")))[thousand_indices]
    x_test = pad_sequences(x_data[thousand_indices], maxlen=512, dtype='float',
        padding='post')

    model = load_model('models/deepcnn.hdf5')

    attributions, summed_attributions = create_explanations(model, 
        x_test[thousand_indices], y_test[thousand_indices])

    with open('correlation.txt', 'w') as correlation_file:
        for key1, key2 in combinations(attributions.keys(), 2):
            correlation_file.write(f'2d correlation between {key1} and {key2}:'
            correlation_file.write(explanation_correlation(attributions[key1],
                attributions[key2], thousand_indices))
            correlation_file.write('')

            correlation_file.write(f'1d correlation between {key1} and {key2}:'
            correlation_file.write(explanation_correlation(summed_attributions\
                [key1], summed_attributions[key2], thousand_indices))
            correlation_file.write('')


if __name__ == "__main__":
    main()
