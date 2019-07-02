# xsurvey

Code for Individual Module "A Survey of Explainability Methods for Neural Network Classifiers in Natural Language Processing"

## Getting Started

* create & activate environment and `pip install -r requirements.txt`

* install DeepExplain `pip install -e git+https://github.com/marcoancona/DeepExplain.git#egg=deepexplain`

* clone Repository

* copy `sample.config.ini` to `config.ini`

* set folders in `config.ini`

* create folders with `python create_folders.py`

* download dataset from https://www.kaggle.com/utathya/imdb-review-dataset and place it in the `dataset` folder specified

* download and unpack GloVe from http://nlp.stanford.edu/data/glove.840B.300d.zip and place it in the `glove` folder specified

## Run code

* `python preprocess.py`

* `python dist_plot.py`

* `python train_model.py`

* `python evaluation.py`

* `python plot.py -p -l`, with `-p` being the option for creating plots and `-l` the option for creating colored tex files

* `python correlation.py`
