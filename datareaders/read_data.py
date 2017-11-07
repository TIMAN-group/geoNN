import numpy as np
np.random.seed(1337)
import random
random.seed(1337)
import time
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from data_utils import makeY_coordinates, load_data_matrix, read_user_location, load_data_matrices_regions, load_data_matrices, load_tf_datafile, get_userids, get_text
from keras.preprocessing import sequence

#using ldaformated data (word:count), we create text features ('binary','TF', or 'tfidf')
def load_data_counts_sklearn(train_file, dev_file, test_file, user_locations_file, max_features, coordinates_index=1, line_index=2, mode='tfidf', useVocab=False, regions=False):
    vocab={}
    vocab_index={}
    user_text_train,user_coordinates_train, vocab = load_tf_datafile(train_file, vocab, vocab_index, coordinates_index, line_index)
    user_text_dev,user_coordinates_dev, vocab = load_tf_datafile(dev_file, vocab, vocab_index, coordinates_index, line_index)
    user_text_test,user_coordinates_test, vocab = load_tf_datafile(test_file, vocab, vocab_index, coordinates_index, line_index)
    sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)
    labels = []
    if(not user_locations_file):
        print("Using coordinates")
        X_train, y_train = load_data_matrix(user_coordinates_train.keys(), user_text_train, user_coordinates_train)
        X_dev, y_dev = load_data_matrix(user_coordinates_dev.keys(), user_text_dev, user_coordinates_dev)
        X_test, y_test = load_data_matrix(user_coordinates_test.keys(), user_text_test, user_coordinates_test)
    else:
        user_loc = read_user_location(user_locations_file)
        if(regions):
            print("Using regions")
            X_train, y_train, labels = load_data_matrices_regions(user_coordinates_train.keys(), user_text_train, user_loc, labels)
            X_dev, y_dev, labels = load_data_matrices_regions(user_coordinates_dev.keys(), user_text_dev, user_loc, labels)
            X_test, y_test, labels = load_data_matrices_regions(user_coordinates_test.keys(), user_text_test, user_loc, labels)
        else:
            print("Using states")
            X_train, y_train, labels = load_data_matrices(user_coordinates_train.keys(), user_text_train, user_loc, labels)
            X_dev, y_dev, labels = load_data_matrices(user_coordinates_dev.keys(), user_text_dev, user_loc, labels)
            X_test, y_test, labels = load_data_matrices(user_coordinates_test.keys(), user_text_test, user_loc, labels)
    vocabulary = None
    if(useVocab):
        vocabulary = sorted_vocab[0:max_features]
    if(mode == 'tf'):
        print("Using tf")
        vectorizer = CountVectorizer(min_df=1,max_features=max_features, stop_words='english', vocabulary = vocabulary, ngram_range=(1,1))
    elif(mode == 'binary'):
        vectorizer = CountVectorizer(min_df=1, max_features=max_features, stop_words='english', vocabulary=vocabulary, ngram_range=(1, 1), binary=True)
    else: #(mode == 'tfidf'):
        print("Using tfidf")
        vectorizer = TfidfVectorizer(min_df=1, max_features=max_features, stop_words='english', vocabulary=vocabulary,
                                     ngram_range=(1, 1), sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train)
    X_dev = vectorizer.transform(X_dev)
    X_test = vectorizer.transform(X_test)
    vocab = {key:value+1   for key,value in vectorizer.vocabulary_.items()}
    return (X_train, y_train), (X_dev, y_dev), (X_test, y_test), labels, vocab


def files(HOME_DIR, DATA_DIR, data):
    print(data)
    coordinates_index = 1
    line_index = 2
    full = None  #HOME_DIR + DATA_DIR + '/wing-baldridge-2011/twitter-geotext/GeoText.2010-10-12/full_text-fixed.txt' and #HOME_DIR + DATA_DIR + '/wing-baldridge-2014/twitter-us/full/full.txt'
    regions = False
    user_locations_file = None
    #default ==> GEOTEXT_COORDINATES
    train_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2011/twitter-geotext/docthresh-0/twitter-geotext-docthresh-0-training.data.txt.bz2'
    dev_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2011/twitter-geotext/docthresh-0/twitter-geotext-docthresh-0-dev.data.txt.bz2'
    test_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2011/twitter-geotext/docthresh-0/twitter-geotext-docthresh-0-test.data.txt.bz2'
    if ('GEOTEXT_STATES' in data):
        # states classification
        user_locations_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2011/twitter-geotext/GeoText.2010-10-12/eisenstein_locations.csv'
    elif ('GEOTEXT_REGIONS' in data):
        # regions classification
        regions = True
        user_locations_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2011/twitter-geotext/GeoText.2010-10-12/eisenstein_locations.csv'
    elif ('TWUS' in data):
        train_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2014/twitter-us/gutonly-big-training.data.txt.bz2'
        dev_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2014/twitter-us/gutonly-big-dev.data.txt'
        test_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2014/twitter-us/gutonly-big-test.data.txt.bz2'
    elif (data == 'TWWORLD'):
        line_index = 3
        full = None
        train_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2014/twitter-world/twitter-world-training.data.txt'
        dev_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2014/twitter-world/twitter-world-dev.data.txt'
        test_file = HOME_DIR + DATA_DIR + '/wing-baldridge-2014/twitter-world/twitter-world-test.data.txt'
    return train_file, dev_file, test_file, user_locations_file, full, regions, coordinates_index, line_index

def read(data, max_features, mode, HOME_DIR = os.path.expanduser('~'), DATA_DIR = '/scratch/web.corral.tacc.utexas.edu/utcompling', useVocab=False):
    print('[read_data.py]: Loading data...')
    X_train_text = None
    X_dev_text = None
    X_test_text = None
    train_file, dev_file, test_file, user_locations_file, full, regions, coordinates_index, line_index = files(HOME_DIR,DATA_DIR,data)
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test), labels, sorted_vocab = load_data_counts_sklearn(
        train_file, dev_file, test_file, user_locations_file, max_features, coordinates_index=coordinates_index,
        line_index=line_index, regions=regions, mode=mode, useVocab=useVocab)
    max_features = X_train.shape[1]
    if (user_locations_file):
        output_dim = len(set(labels))
        #y_train, y_dev, y_test = makeY_state(y_train, y_dev, y_test, output_dim)
        y_train, y_dev, y_test = makeY_coordinates(y_train, y_dev, y_test)
    else:
        output_dim = 2
        y_train, y_dev, y_test = makeY_coordinates(y_train, y_dev, y_test)

    print('[read_data.py]: mode:{0}, max_feature:{1}'.format(mode, max_features))
    dataset = {'data': data, 'X_train': X_train, 'y_train': y_train, 'X_dev': X_dev, 'y_dev': y_dev, 'X_test': X_test,
               'y_test': y_test, 'max_features': max_features,
               'sorted_vocab': sorted_vocab, 'labels': labels,
               'X_train_text': X_train_text,
               'X_dev_text': X_dev_text, 'X_test_text': X_test_text, 'output_dim':output_dim}
    return dataset
