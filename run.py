import numpy as np
np.random.seed(1337)
import random
random.seed(1337)
import os
os.environ['KERAS_BACKEND'] = 'theano' #'tensorflow'
import time
from datareaders.data_utils import  batch_generator
import sys
import argparse
import traceback
from models.models import geoMLP
from models.models_utils import  Train, print_results
from datareaders.read_data import read

def run(dataset, dropout_U, dense_size, batchnorm, activation, lr, resultsfile=None, nb_epoch=1, batch_size=64, samples=64000, n_layers=3):
    print("[run.py]: X_train", dataset['X_train'].shape)
    print("[run.py]: X_dev", dataset['X_dev'].shape)
    print("[run.py]: X_test", dataset['X_test'].shape)
    print("[run.py]: y_train", dataset['y_train'].shape)
    print("[run.py]: y_dev", dataset['y_dev'].shape)
    print("[run.py]: y_test", dataset['y_test'].shape)
    print("[run.py]: X_train", type(dataset['X_train']))
    print("[run.py]: X_dev", type(dataset['X_dev']))
    print("[run.py]: X_test", type(dataset['X_test']))
    print("[run.py]: y_train", type(dataset['y_train']))
    print("[run.py]: y_dev", type(dataset['y_dev']))
    print("[run.py]: y_test", type(dataset['y_test']))

    print("[run.py]: y_train", dataset['y_train'])


    samples = min(samples, dataset['X_train'].shape[0])
    output_dim =  dataset['output_dim']
    print('[run.py]: Build model...')
    folder = 'results/'
    if not os.path.exists(folder):
        os.makedirs(folder)
    modelstring = folder +  dataset['data'] + '_' + str(dense_size) + '_' + str(dropout_U) + '_' + str(output_dim) + '_' + activation + '_' + str(batchnorm) + '_' + str(lr) + 'n_layers' + str(n_layers)
    print('[run.py]: model string:', modelstring)
    resultstring = 'Build MLP model with max_features:{0}, dense_size:{1}, dropout_U:{2}, output_dim:{3},  activation:{4}, batchnorm:{5}, lr:{6}, n_layers:{7}, batch_size:{8}'\
        .format(dataset['max_features'], dense_size, dropout_U, output_dim, activation, batchnorm, lr, n_layers, batch_size)
    print('[run.py]: resultstring string:', resultstring)
    geoModel, early_stopping, checkpoint = geoMLP(dataset['max_features'], dense_size, dropout_U, output_dim, activation, batchnorm,lr, n_layers, modelstring)
    if (resultsfile):
        target = open(resultsfile, 'a')
        target.write(resultstring + '\n')
        target.close()
    history = Train(dataset['X_train'], dataset['y_train'], dataset['X_dev'], dataset['y_dev'], geoModel, batch_size, nb_epoch, samples_per_epoch=samples,
                        callback=[early_stopping, checkpoint],  batch_generator=batch_generator)
    geoModel.save_weights(modelstring + ".h5py")
    score = print_results(dataset, output_dim, resultsfile, geoModel, batch_size, best=False)
    if (os.path.exists(modelstring + "_best.h5py")):
        geoModel.load_weights(modelstring + "_best.h5py")
        score = print_results(dataset, output_dim, resultsfile, geoModel, batch_size, best=True)
    return geoModel, score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to do experiments on different datasets.', prefix_chars='-+')
    parser.add_argument('--dset', dest='dset', default='GEOTEXT_COORDINATES',
                        help='choose dataset GEOTEXT_STATES, GEOTEXT_REGIONS, GEOTEXT_COORDINATES, TWUS, TWWORLD')
    parser.add_argument('--max_features', dest='max_features', default=50000, type=int, help='choose max number of features')
    parser.add_argument('--resultsfile', dest='resultsfile', default=None, help='write results to file')
    parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, help='choose batch_size')
    parser.add_argument('--nb_epochs', dest='nb_epochs', default=1000, type=int, help='choose batch_size')
    parser.add_argument('--mode', dest='mode', default='tfidf', help='choose between [tfidf, tf or binary')
    parser.add_argument('--samples', dest='samples', default=64000, type=int, help='choose samples_per_epoch')

    parser.add_argument('--dropout', dest='dropout', default=[0, 0.25, 0.5], nargs='+', type=float)
    parser.add_argument('--n_layers', dest='n_layers', default=[3, 5, 10, 20], nargs='+', type=float)
    parser.add_argument('--dense_size', dest='dense_size', default=[128, 512, 1024, 4096], nargs='+', type=int)
    parser.add_argument('--batchnorm', dest='batchnorm', default=[True, False], nargs='+', type=bool)
    parser.add_argument('--activation', dest='activation', default=['sigmoid', 'prelu', 'relu'], nargs='+')
    parser.add_argument('--lr', dest='lr', default=[0.1, 0.01, 0.001, 0.0001], type=float, nargs='+')
    args = parser.parse_args()
    print('args:', args)
    dataset =  read(args.dset, args.max_features,args.mode)
    nb_epochs = args.nb_epochs
    batch_size = args.batch_size
    start_time = time.time()
    for n_layers in args.n_layers:
        for dropout in args.dropout:
            for dense_size in args.dense_size:
                for batchnorm in args.batchnorm:
                    for activation in args.activation:
                        for lr in args.lr:
                            try:
                                run(dataset, dropout, dense_size, batchnorm, activation, lr,resultsfile = args.resultsfile, nb_epoch = nb_epochs, batch_size = batch_size, n_layers = n_layers, samples=args.samples)
                            except Exception as error:
                                traceback.print_exc(file=sys.stdout)
    end_time = time.time()
    print("Time for fitting all combinations: " + str(end_time - start_time))
