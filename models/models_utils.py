import numpy as np
np.random.seed(1337)
import random
random.seed(1337)
import os
from keras.optimizers import SGD
from loss_functions import scores, errors_mean
import scipy.sparse as sps
from keras.models import model_from_json
from datareaders.data_utils import batch_generator
from models import geoMLP


def Train(X_train, y_train, X_dev, y_dev, geoModel, batch_size, nb_epoch, samples_per_epoch=None, callback=None, batch_generator=None):
    if(sps.issparse(X_dev)):
        X_dev = np.array(X_dev.todense())
    if (sps.issparse(X_train)):
        samples_per_epoch = min(samples_per_epoch, X_train.shape[0])
        history = geoModel.fit_generator(batch_generator(X_train, y_train, batch_size, samples_per_epoch),samples_per_epoch=samples_per_epoch, nb_epoch=nb_epoch, verbose=1, validation_data=(X_dev, y_dev), callbacks=callback)
    else:
        history = geoModel.fit(X_train, y_train, shuffle=True, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_dev, y_dev), callbacks=callback)
    return history

def Test(X_dev, y_dev, X_test, y_test, geoModel, batch_size):
    if(sps.issparse(X_dev)):
        X_dev = np.array(X_dev.todense())
    if (sps.issparse(X_test)):
        X_test = np.array(X_test.todense())
    score_dev = geoModel.evaluate(X_dev, y_dev, batch_size=batch_size, verbose=1)
    y_predicted_dev = geoModel.predict(X_dev, batch_size=batch_size, verbose=1)
    assert len(y_predicted_dev) == len(y_dev)
    score_test = geoModel.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
    y_predicted_test = geoModel.predict(X_test, batch_size=batch_size, verbose=1)
    assert len(y_predicted_test) == len(y_test)
    return y_predicted_dev, score_dev, y_predicted_test, score_test

def saveModel(model, modelfile, weightfile):
    # serialize model to JSON
    model_json = model.to_json()
    with open(modelfile, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weightfile, overwrite=True)
    print("Saved model to disk")

def loadModel(modelstring, lr, output_dim):
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    # load json and create model
    modelfile = modelstring + '_model.json'
    weightfile = modelstring + "_model.h5"
    print(modelfile)
    print(weightfile)
    json_file = open(modelfile, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    print("Loaded model from disk")
    if(output_dim==2):
        loaded_model.compile(loss=errors_mean, optimizer=sgd)
    else:
        loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    loaded_model.load_weights(weightfile)
    return loaded_model


def print_results(dataset, output_dim, resultsfile, geoModel, batch_size, best=False):
    y_predicted_dev, score_dev, y_predicted_test, score_test = Test(dataset['X_dev'], dataset['y_dev'], dataset['X_test'], dataset['y_test'], geoModel, batch_size)
    if(best):
        dataset['y_predicted_dev'] = y_predicted_dev
        dataset['y_predicted_test'] = y_predicted_test
        dev = 'dev_best:'
        test = 'test_best:'
    else:
        dev = 'dev:'
        test = 'test:'
    result = dev + str(score_dev) + test + str(score_test)
    if (output_dim == 2):
        pred, y_correct, distances, score_dev = scores(y_predicted_dev,  dataset['y_dev'])
        pred, y_correct, distances, score_test = scores(y_predicted_test, dataset['y_test'])
        result = dev + score_dev + test + score_test
    print(result)
    if (resultsfile):
        target = open(resultsfile, 'a')
        target.write(result + '\n')
        target.close()
    return score_dev

