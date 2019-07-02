import numpy as np
from keras import backend
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import sys, getopt
import csv
import matplotlib
import matplotlib.pyplot as plt
import pickle
from configparser import ConfigParser

from explanations import create_explanations
from preprocess import spacyfy


def plot_explanation(attribution, name, dim, images_folder, save_fig=True):
    plt.figure(figsize=(12.8,7.5), dpi=50)
    data_array = attribution
    plt.title(f'DeepExplain {dim}: {name}')

    if dim = '2d':
        data_max = np.max(np.abs(data_array))
        plt.imshow(data_array, cmap='seismic', vmin=-data_max, vmax=data_max)
        plt.colorbar()
        plt.xlabel('index of token')
        plt.ylabel('word vector dimensions')
    elif dim = '1d':            
        mask1 = data_array < -1e-8
        mask2 = data_array > 1e-8
        plt.bar(np.arange(512)[mask1], data_array[mask1], color = 'blue')
        plt.bar(np.arange(512)[mask2], data_array[mask2], color = 'red')
        plt.xlabel('index of token')
        plt.ylabel('summed attribution score')

    if save_fig:
        filename = f'{images_folder}/de{dim}_{name}.eps'
        plt.savefig(filename)
        print(f'saved plot {filename}')

def find_line(imdb_file, index):
    with open(imdb_file, encoding="latin-1") as csvFile:
        reader = csv.reader(csvFile)
        count=0
        for row in reader:
            if row[1] == 'test':
                if count==index:
                    return row[2]
                else:
                    count += 1

def text_to_latex(line, scores):
    output_list = []
    for token in spacyfy(line):
        output_list.append(token.lower_)

    output_text = ''
    
    for word, score in zip(output_list, scores):
        red = 255*(1+min(score, 0))
        green = 255*min(1-score, 1+score)
        blue = 255*(1-max(score, 0))
        output_text += '\colorbox[RGB]{' + str(int(red)) + ',' + \
            str(int(green)) + ',' + str(int(blue)) + '}{\strut ' + word + '} '
        
    return output_text

def main():

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'pl')
    except getopt.GetoptError as err:
        print(err)
        sys.exit()

    print(backend.tensorflow_backend._get_available_gpus())

    config = ConfigParser()
    config.read('config.ini')

    dataset_folder = config.get('DATA FOLDER', 'dataset')
    images_folder = config.get('DATA FOLDER', 'images')
    latex_folder = config.get('DATA FOLDER', 'latex')
    preprocessed_folder = config.get('DATA FOLDER', 'preprocessed')
    data_index = config.get('PLOT', 'data_index')

    matplotlib.rcParams.update({'font.size': 20})

    x_data = pickle.load(open(f'{preprocessed_folder}/x_test.p', 'rb'))
    x_test = pad_sequences(x_data[data_index:data_index+1], maxlen=512,
        dtype='float', padding='post')
    y_test = np.array(pickle.load(open(f'{preprocessed_folder}/y_test.p',
        'rb')))[data_index:data_index+1]

    model = load_model('models/deepcnn.hdf5')

    attributions, attributions_summed = create_explanations(model, x_test,
        y_test)

    for p,l in opts:
        if p in ("-p"):
            for key in attributions.keys():
                plot_explanation(np.transpose(attributions[key][0]), key, '2d',
                    images_folder)
                plot_explanation(attributions_summed[key][0], key, '1d',
                    images_folder)
        elif l in ("-l"):
            latex = dict()
            for key in attributions_summed.keys():
                latex[key] = text_to_latex(find_line(f'{dataset_folder}/' + \
                    'imdb_master.csv', data_index), attributions_summed[key] \
                    [0]/np.max(np.abs(attributions_summed[key][0])))
                with open(f'{latex_folder}/{key}_{data_index}.tex', 'w') as 
                    tex_file: tex_file.write(latex[key])
        else:
            sys.exit()


if __name__ == "__main__":
    main()
