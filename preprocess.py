import numpy as np
import spacy
import csv
from collections import defaultdict
from configparser import ConfigParser


def glove_to_dict(glove_file):
    glove_dict = defaultdict(lambda: np.zeros(300))
    
    with open(glove_file) as csvFile:
        reader = csv.reader(csvFile, delimiter=' ', quoting=csv.QUOTE_NONE)
        for row in reader:
            values = np.array(row[1:])
            glove_dict[row[0]] = values.astype(np.float)
    return glove_dict

def spacyfy(line):
    line2 = line.replace('<br />', ' ')
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(line2)
    return doc

def preprocess(line, glove_dict):
    output_list = []
    doc = spacyfy(line)
    for token in doc:
        output_list.append(glove_dict[token.lower_])
    return np.array(output_list)

def dataloader(csv_file, glove_dict):
    x_train, y_train, x_test, y_test = [], [], [], []
    count = 0
    
    with open(csv_file, encoding="latin-1") as csvFile:
        reader = csv.reader(csvFile)
        
        for row in reader:
            if row[1] == 'train':
                if row[3] == 'neg':
                    y_train.append(np.array([1, 0]))
                elif row[3] == 'pos':
                    y_train.append(np.array([0, 1]))
                else:
                    continue
                x_train.append(preprocess(row[2], glove_dict))
                
            elif row[1] == 'test':
                if row[3] == 'neg':
                    y_test.append(np.array([1, 0]))
                elif row[3] == 'pos':
                    y_test.append(np.array([0, 1]))
                else:
                    continue
                x_test.append(preprocess(row[2], glove_dict))
            count += 1
            if count % 100 == 0:
                print(count)
    
    return x_train, y_train, x_test, y_test

def main():        
    config = ConfigParser()
    config.read('config.ini')

    dataset_folder = config.get('DATA FOLDER', 'dataset')
    preprocessed_folder = config.get('DATA FOLDER', 'preprocessed')

    glove_dict = glove_to_dict(f'{dataset_folder}/glove.840B.300d.txt')

    x_train, y_train, x_test, y_test = dataloader(f'{dataset_folder}/imdb_master.csv', glove_dict)

    pickle.dump(x_train, open(f'{preprocessed_folder}/x_train.p', 'wb'))
    pickle.dump(y_train, open(f'{preprocessed_folder}/y_train.p', 'wb'))
    pickle.dump(x_test, open(f'{preprocessed_folder}/x_test.p', 'wb'))
    pickle.dump(y_test, open(f'{preprocessed_folder}/y_test.p', 'wb'))


if __name__ == "__main__":
    main()
