import tensorflow as tf
import numpy as np
import h5py
import os
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import openpyxl
import random
import time
from camelyon_util import *

class samplegen():
    def __init__(self, dirname):
        self.dirname = dirname
    def __call__(self):
        with h5py.File(self.dirname, 'r') as f:
            for i in range(f['bag'].shape[0]):
                yield f['bag'][i], f['label']


        
def intelsampling(datanamelist, input_shape, encoded_shape, disc_model, encoder, embedding_dir, train, sampling=False):
    datadict = {}
    scoredict = {}
    coorddict = {}
    labeldict = {}
    bsize = 50
    datanamelist = sorted(datanamelist)
    for idx, name in enumerate(datanamelist):
        print(name)
        if sampling:
            scores=np.empty((0,1),dtype=np.float64)
        bag=np.empty((0,encoded_shape))
        
        start = time.time()
        dirname = name
        gen = samplegen(name)

        ds_test=tf.data.Dataset.from_generator(generator=gen, output_types=(tf.int8, tf.int32),\
                                                    output_shapes=(tf.TensorShape([3, input_shape, input_shape]),tf.TensorShape([])))\
                .map(load_discimage, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                .batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
                
        for x,y in ds_test:
            if sampling:
                scores.resize((scores.shape[0]+x.shape[0],1))
                scores[-x.shape[0]:]=disc_model.predict_on_batch(x)
            bag.resize((bag.shape[0]+x.shape[0],encoded_shape))
            bag[-x.shape[0]:]=encoder.predict_on_batch(x)
            
        with h5py.File(name, 'r') as f:
            label=f['label'][()]
            # coords = f['coords'][:]
            
        print(bag.shape)
        # print(coords.shape)
        
        if bag.shape[-1]!=encoded_shape:
            raise IndexError('wrong shape of bag')
        
        datadict[name]=bag
        labeldict[name]=label
        if sampling:
            scoredict[name]=scores
        # coorddict[name]=coords
        end = time.time()
        print('Used {} s'.format(end-start))
        
    np.save(embedding_dir+train+'embeddeddata.npy',datadict)
    # np.save(embedding_dir+train+'coords.npy',coorddict)
    np.save(embedding_dir+train+'labels.npy',labeldict)
    if sampling:
        np.save(embedding_dir+train+'scores.npy',scoredict)

    
def createbags_final(datanamelist, scoreset, dataset, labelset, K, encoded_shape):
    '''
    select top, bot-k% instances
    '''
    bags = []
    labels = []
    for idx, name in enumerate(datanamelist):
        print(name)
        scores = scoreset[name]
        scores = np.squeeze(scores, 1)
        sortidxs = np.argsort(-scores)
        length = scores.shape[0]
            
        embedding1 = dataset[name][scores>=0.5]
        embedding0 = dataset[name][scores<0.5]
        
        scores1 = scores[scores>=0.5]
        sortidxs1 = np.argsort(-scores1)
        embedding1 = embedding1[sortidxs1[:round(K*scores1.shape[0])]]

        scores0 = scores[scores<0.5]
        sortidxs0 = np.argsort(-scores0)
        embedding0 = embedding0[sortidxs0[-round(K*scores0.shape[0]):]]
        embedding = np.concatenate((embedding1, embedding0), axis=0)
    
        
        bags.append(embedding)
        labels.append(int(labelset[name]))
    return bags, labels
    


def trainingname_batch_split(normalnamelist, tumornamelist, normalbin, tumorbin):
    random.shuffle(normalnamelist)
    random.shuffle(tumornamelist)
    datanamelist = []
    for i, j in zip(range(0, len(normalnamelist), normalbin), range(0, len(tumornamelist), tumorbin)):
        datanamelist.extend(normalnamelist[i:i+normalbin])
        datanamelist.extend(tumornamelist[j:j+tumorbin])
    
    return datanamelist
        
        
        
        
        