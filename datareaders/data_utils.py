import numpy as np
np.random.seed(1337)
import random
random.seed(1337)
import bz2
import time
from collections import OrderedDict, Counter, defaultdict
from sklearn.feature_selection import SelectKBest,chi2
from keras.utils import np_utils
from keras.engine.training import _make_batches,_slice_arrays as _slice_X
import scipy.sparse as sps
from keras import backend as K


def batch_generator(X, y, batch_size, samples_per_epoch):
    while 1:
        index_array = np.arange(X.shape[0])
        np.random.shuffle(index_array)
        batches = _make_batches(samples_per_epoch, batch_size)
        # print("\n",index_array[0:2])
        # print("=====================")
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            # print("\n", batch_start, batch_end, batch_index, len(batch_ids))
            X_batch = _slice_X(X, batch_ids)
            y_batch = _slice_X(y, batch_ids)
            if (sps.issparse(X)):
                X_batch = np.array(X_batch.todense())
            else:
                X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            yield (X_batch, y_batch)

def makeY_coordinates(y_train, y_dev, y_test):
    y_train = np.array(y_train)
    y_train = y_train.astype(np.float)
    y_dev = np.array(y_dev)
    y_dev = y_dev.astype(np.float)
    y_test = np.array(y_test)
    y_test = y_test.astype(np.float)
    return y_train, y_dev, y_test


def makeY_state(y_train, y_dev, y_test, dim):
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, dim)
    y_dev = np_utils.to_categorical(y_dev, dim)
    y_test = np_utils.to_categorical(y_test, dim)
    return y_train, y_dev, y_test

def get_activations(model, layer, X_batch):
    if(sps.issparse(X_batch)):
        X_batch = np.array(X_batch.todense())
    else: 
        X_batch = np.array(X_batch)
    get_activations = K.function([model.layers[0].input, K.learning_phase()], model.layers[layer].output)
    activations = get_activations([X_batch,0])
    return activations

def load_tf_datafile(dfile, vocab, vocab_index, coordinates_index=1, line_index=2):
    try:
        users = bz2.BZ2File(dfile, 'r').readlines()
    except IOError:
        users = open(dfile, 'r').readlines()
    user_text_seq_full = {}
    user_coordinates = {}
    start = time.time()
    for i in range(len(users)):
        try:
            user = users[i].split('\t')
            uname = user[0]
            coord = map(lambda x :float(x), user[coordinates_index].split(','))
            content = user[line_index].split()
            w_dict, out_str  = list_to_dict(content, vocab)
            #users[i] = (uname, coord, w_dict, out_str)
            if(uname in user_text_seq_full):
                print("Problem! User already exists!")
            user_text_seq_full[uname] = out_str
            user_coordinates[uname] = coord
            if(i%100000==0):
                end = time.time()
                print('Finished {} out of {} in {}'.format(i,len(users), str(end - start)))
                start = time.time()
        except Exception as error:
            print(error)
            #print(users)
            #print(uname)
            #print(user[coordinates_index])
            #print(user[line_index])
    #for i, word in enumerate(vocab.keys()):
        #vocab_index[word] = i
    #print(users)
    #print(vocab)
    #print(vocab_index)
    return user_text_seq_full,user_coordinates, vocab
    
def list_to_dict(l, vocab):
    out_str = ''
    dic = {}
    for item in l:
        item = item.split(':')
        w = item[0]
        c = int(item[1])
        dic[w] = c
        vocab[w] = vocab.get(w, 0)+c
        out_str +=  (w+' ') * int(c)
        """for x in range(0, int(c)):
            out_str += w
            out_str += ' '"""
    return dic, out_str

def get_userids(dataset, vocab, train=False):
    userids = []
    print(bz2.__file__)
    print(dataset)
    print(type(dataset))
    try:
        users = bz2.BZ2File(dataset, 'r').readlines()
    except IOError:
        users = open(dataset, 'r').readlines()
    #with bz2.open(dataset, 'rt') as f:
    for i in range(len(users)):
        line = users[i]
        #print(line)
        content = line.split('\t')
        userid =  content[0] 
        userids.append(userid)
        if(train):
            text = content[2]
            words = text.split(' ')
            for w_c in words:
                w_c = w_c.rsplit(':',1)
                w = w_c[0]
                c = w_c[1]
                if not w in vocab:
                    vocab[w] = int(c)
    #f.close()
    return userids, vocab

def get_text(dataset, vocab=None, padding=False):
    user_text_seq_full = {}
    user_coordinates = {}
    with open(dataset, 'r') as f:
        for line in f:
            content = line.split('\t')
            userid =  content[0]
            lat =  content[3]
            lon = content[4]
            if userid not in user_text_seq_full:
                user_text_seq_full[userid] = ""
                user_coordinates[userid] = [lat, lon]
            text = content[5]
            if padding:
                user_text_seq_full[userid]+=(text)+" #ENDOFTWEET #ENDOFTWEET #ENDOFTWEET #ENDOFTWEET #ENDOFTWEET "
            else:
                user_text_seq_full[userid]+=(text)+" "
    f.close()
    return user_text_seq_full, user_coordinates

def load_data_matrix(userids, user_text_seq_full, user_coordinates):
    y = []
    X = []
    for user in userids:
        X.append(user_text_seq_full[user])
        y.append(user_coordinates[user])
    return X,y

def load_data_matrices(userids, user_text_seq_full, user_loc, labels):
    y = []
    X = []
    for user in userids:
        user_loc[user][0]
        if 'UKN' not in user_loc[user][0]:
            if 'District of Columbia' in user_loc[user][2]:
                if user_loc[user][0] == 'United States of America':
                    X.append(user_text_seq_full[user])
                    if not user_loc[user][2]in labels:
                        labels.append(user_loc[user][2])
                    y.append(labels.index(user_loc[user][2]))
        if 'UKN' not in user_loc[user][1]:
            if user_loc[user][0] == 'United States of America':
                X.append(user_text_seq_full[user])
                if not user_loc[user][1]in labels:
                    labels.append(user_loc[user][1])
                y.append(labels.index(user_loc[user][1]))
    #shortlist = labels
    #print(len(labels))
    return X,y,labels
    
def load_data_matrices_regions(userids, user_text_seq_full, user_loc, labels):    
    regions = {}
    regions['Connecticut'] = 0
    regions['Maine'] = 0
    regions['Massachusetts'] = 0
    regions['New Hampshire'] = 0
    regions['Rhode Island'] = 0
    regions['Vermont'] = 0
    regions['New Jersey'] = 0
    regions['New York'] = 0
    regions['Pennsylvania'] = 0
    regions['Indiana'] = 1
    regions['Illinois'] = 1
    regions['Michigan'] = 1
    regions['Ohio'] = 1
    regions['Wisconsin'] = 1
    regions['Iowa'] = 1
    regions['Kansas'] = 1
    regions['Minnesota'] = 1
    regions['Missouri'] = 1
    regions['Nebraska'] = 1
    regions['North Dakota'] = 1
    regions['South Dakota'] = 1
    regions['Delaware'] = 2
    regions['District of Columbia'] = 2
    regions['Florida'] = 2
    regions['Georgia'] = 2
    regions['Maryland'] = 2
    regions['North Carolina'] = 2
    regions['South Carolina'] = 2
    regions['Virginia'] = 2
    regions['West Virginia'] = 2
    regions['Alabama'] = 2
    regions['Kentucky'] = 2
    regions['Mississippi'] = 2
    regions['Tennessee'] = 2
    regions['Arkansas'] = 2
    regions['Louisiana'] = 2
    regions['Oklahoma'] = 2
    regions['Texas'] = 2
    regions['Arizona'] = 3
    regions['Colorado'] = 3
    regions['Idaho'] = 3
    regions['New Mexico'] = 3
    regions['Montana'] = 3
    regions['Utah'] = 3
    regions['Nevada'] = 3
    regions['Wyoming'] = 3
    regions['California'] = 3
    regions['Oregon'] = 3
    regions['Washington'] = 3
    
    y = []
    X = []
    for user in userids:
        user_loc[user][0]
        if 'UKN' not in user_loc[user][0]:
            if 'District of Columbia' in user_loc[user][2]:
                if user_loc[user][0] == 'United States of America':
                    region = regions[user_loc[user][2]]
                    #print(region, user_loc[user][2])
                    if not region in labels:
                        labels.append(region)
                    X.append(user_text_seq_full[user])
                    y.append(labels.index(region))
        if 'UKN' not in user_loc[user][1]:
            if user_loc[user][0] == 'United States of America':
                region = regions[user_loc[user][1]]
                #print(region, user_loc[user][1])
                if not region in labels:
                    labels.append(region)
                y.append(labels.index(region))
                X.append(user_text_seq_full[user])
    #shortlist = labels
    #print(len(labels))
    return X,y,labels

def read_user_location(dataset): #"einstein_locations.csv"
    user_locations = {}
    with open(dataset, 'r') as f:
        i=0
        for line in f:
            if(i>0):
                content = line.split(',')
                userid =  content[0] 
                country = content[1]
                state =  content[2]
                county = content[3]
                city = content[4]
                user_locations[userid] = [country,state,county,city] 
            i+=1
    f.close()
    return user_locations

