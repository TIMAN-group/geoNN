import numpy as np
np.random.seed(1337)
import random
random.seed(1337)
import theano.tensor as T

#all evaluation metrics printed
def scores(y_predicted, y_correct):
    dists  = dist(y_predicted, y_correct)
    d = dists.eval()
    if(not np.isnan(d).any()):
        print("mean_dist_km: %f" % np.mean(d))
        print("med_dist_km: %f" % np.median(d))
        print("acc_161: %f" % ((d<=161).sum()/float(len(d.flat))))
        print("len(d.flat):",len(d.flat))
        print("(d<=161).sum():",(d<=161).sum())
        result = 'mean_dist_km:' + str(np.mean(d)) + ' med_dist_km:' + str(np.median(d))+ ' acc161:' + str(((d<=161).sum()/float(len(d.flat))))
    else:
        result = 'mean_dist_km:nan  med_dist_km:nan acc161:nan'
    return y_predicted, y_correct, d, result

#A helper method that computes distance between two points on the surface of earth according to their coordinates.
#Inputs are tensors.
def dist(y_pred, y):	
    y_pred_ra = T.deg2rad(y_pred)
    y_ra = T.deg2rad(y)
    lat1 = y_pred_ra[:, 0]
    lat2 = y_ra[:, 0]
    dlon = (y_pred_ra - y_ra)[:, 1]
    EARTH_R = 6372.8
    y = T.sqrt((T.cos(lat2) * T.sin(dlon)) ** 2+ (T.cos(lat1) * T.sin(lat2) - T.sin(lat1) * T.cos(lat2) * T.cos(dlon)) ** 2)
    x = T.sin(lat1) * T.sin(lat2) + T.cos(lat1) * T.cos(lat2) * T.cos(dlon)
    c = T.arctan2(y, x)
    return EARTH_R * c

def errors_mean(y_true, y_pred):
    if y_true.ndim != y_pred.ndim:
        raise TypeError('y should have the same shape as self.y_pred',('y', y_true.type, 'y_pred', y_pred.type))
    print("y_true.dtype",y_true.dtype)
    if str(y_true.dtype).startswith('float'):
        dists = dist(y_pred, y_true)
        return T.mean(dists)
    else:
        raise NotImplementedError()
