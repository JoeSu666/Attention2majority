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
from util import *

def intelsampling(datanamelist, extestnamelist, encoded_shape, disc_model, encoder, datapath, embedding_dir, start):
    dataset={}
    coordsset={}
    labelset={}


    for idx, name in enumerate(datanamelist+extestnamelist):
        with h5py.File(datapath+name,'r') as f:
            label=f['label'][0][0]-1
            length=f['bag'].shape[0]
            predictions=np.empty((0,),dtype=np.int8)
            scores=np.empty((0,),dtype=np.float64)
            bag=np.empty((0,encoded_shape))
            coords=np.empty((0,2))
            print('============================')
            print('encoding slide{}'.format(idx))

            for i in chunks(list(range(length)),50):
                patch=load_image(f['bag'][i])
                predictions.resize((predictions.shape[0]+len(i),))
                predictions[-len(i):]=np.argmax(disc_model.predict(patch),1)
                scores.resize((scores.shape[0]+len(i),))
                scores[-len(i):]=np.max(disc_model.predict(patch),1)
            print('finish grading')
            mv=np.argsort(-np.bincount(predictions,minlength=3))
            sortidxs=np.argsort(-scores)
            predictions=predictions[sortidxs]

            if predictions.shape[0]!=length:
                raise IndexError('score not match patch')

            sortidx=np.empty((0,),dtype=np.int8)
            for i in mv:
                temp=np.where(predictions==i)[0]
                sortidx=np.concatenate((sortidx,sortidxs[temp]),axis=0)

            if length<2000:
                for i in sortidx:
                    patch=load_image(f['bag'][i])
                    bag.resize((bag.shape[0]+1,encoded_shape))
                    bag[-1]=encoder.predict(patch)
                    coords.resize((coords.shape[0]+1,2))
                    coords[-1]=f['coords'][:,i]

            else:
                for i in sortidx[:2000]:
                    patch=load_image(f['bag'][i])
                    bag.resize((bag.shape[0]+1,encoded_shape))
                    bag[-1]=encoder.predict(patch)
                    coords.resize((coords.shape[0]+1,2))
                    coords[-1]=f['coords'][:,i]
            print(bag.shape)
            print(bag.shape[-1])
            if bag.shape[-1]!=encoded_shape:
                raise IndexError('wrong shape of bag')
            dataset[name]=bag
            coordsset[name]=[coords,predictions[np.where(sortidxs==sortidx)[0]][:2000],sortidx]
            labelset[name]=label

    print('finish')
    np.save(embedding_dir+str(start)+'fold'+'embeddeddata.npy',dataset)
    np.save(embedding_dir+str(start)+'fold'+'coords.npy',coordsset)
    np.save(embedding_dir+str(start)+'fold'+'labels.npy',labelset)