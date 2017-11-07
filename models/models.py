import numpy as np
np.random.seed(1337)
import random
random.seed(1337)
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers.embeddings import Embedding
from keras.layers.advanced_activations import PReLU
from loss_functions import errors_mean
from keras.callbacks import EarlyStopping,  ModelCheckpoint

def geoMLP(max_features, dense_size, dropout, output_dim, activation, batchnorm, lr, n_layers, modelstring):
    model = Sequential()
    if(activation=='prelu' or activation=='relu'):
        kernel_initializer = 'he_normal'
    else:
        kernel_initializer = 'glorot_uniform'
    for i in range(0,n_layers):
        model.add(Dense(dense_size, input_shape=(max_features,), kernel_initializer=kernel_initializer))
        if(batchnorm):
            model.add(BatchNormalization())
        if(activation=='prelu'):
            model.add(PReLU())
        else:
            model.add(Activation(activation))
        model.add(Dropout(dropout))
    model.add(Dense(output_dim, kernel_initializer=kernel_initializer))
    if(batchnorm):
        model.add(BatchNormalization())
    if(output_dim==2):
        model.add(Activation('linear'))
        sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=errors_mean,optimizer=sgd)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
        checkpoint = ModelCheckpoint(modelstring + "_best.h5py", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    else:
        model.add(Activation('softmax'))
        sgd = Adagrad(lr=lr, epsilon=1e-08, decay=0.0)
        model.compile(loss='sparse_categorical_crossentropy',optimizer=sgd,  metrics=["accuracy"])
        early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
        checkpoint = ModelCheckpoint(modelstring + "_best.h5py", monitor='val_acc', verbose=1, save_best_only=True,mode='max')
    return model, early_stopping, checkpoint

