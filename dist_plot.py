import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from configparser import ConfigParser


def main():
    config = ConfigParser()
    config.read('config.ini')
    preprocessed_folder = config.get('DATA FOLDER', 'preprocessed')

    x_train = pickle.load(open(f'{preprocessed_folder}/x_train.p', 'rb'))
    x_train_len_list = [train_array.shape[0] for train_array in x_train]

    plt.xlim(0, 2000)
    sns.distplot(x_train_len_list, bins=[100*idx for idx in range(20)], 
        kde=False, hist_kws={'edgecolor':'black'}).figure.savefig(
        'images/dist.eps', format='eps')


if __name__ == "__main__":
    main()
