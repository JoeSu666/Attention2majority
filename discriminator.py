import tensorflow as tf
import numpy as np
import h5py
import os
import glob
from tensorflow.keras import datasets,layers,models
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import random
import matplotlib
import intelsampling
from camelyon_util import *

tumordir = './datasets/camelyon_1level224/tumor/'
normaldir = './datasets/camelyon_1level224/normal/'
testdir = './datasets/camelyon_1level224/test/'
sampleset_name = './datasets/camelyon_1level224/sampleset.h5'

def random_sampling(datapath, f, ssize, datasetname, group):

    namelist = os.listdir(datapath)
    for count, name in enumerate(namelist):
        count += 1
        if count%10 == 0:
            print('{}/{}'.format(count, len(namelist)))
        with h5py.File(datapath+name,'r') as f1:     
            if f1['bag'].shape[0]<ssize:
                data_size=f1['bag'].shape[0]
                idx=np.random.choice(f1['bag'].shape[0],size=f1['bag'].shape[0],replace=False)
            else:
                data_size=ssize
                idx=np.random.choice(f1['bag'].shape[0],size=data_size,replace=False)
            patches = f1['bag'][:][idx]
            if group == 'test':
                label = f1['label'][()] 
            else:
                label = group
            f[datasetname+'_patches'].resize((f[datasetname+'_patches'].shape[0]+patches.shape[0]),axis=0)
            f[datasetname+'_patches'][-patches.shape[0]:]=patches
            f[datasetname+'_label'].resize((f[datasetname+'_label'].shape[0]+patches.shape[0]),axis=0)
            f[datasetname+'_label'][-patches.shape[0]:]=label

def build_sampleset(namelist, datapath, ssize, savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    for name in namelist:
        if not os.path.isdir(savedir+name):
            os.mkdir(savedir+name)
        group = name.split('_')[0]
        with h5py.File(datapath+group+'/'+name, 'r') as f:
            print(name)
            if f['bag'].shape[0]<ssize:
                data_size=f['bag'].shape[0]
                idx = np.arange(data_size)
#                 idx=np.random.choice(f['bag'].shape[0],size=f['bag'].shape[0],replace=False)
            else:
                data_size=ssize
                idx=np.random.choice(f['bag'].shape[0],size=data_size,replace=False)
            label = f['label'][()]
            print(label)
            for i in range(data_size):
                if i%100 == 0:
                    print('{}/{}'.format(i+1, data_size))
                tmp = {}
                tmp['patch'] = f['bag'][idx[i]]
                tmp['label'] = label
                tmp['coords'] = f['coords'][:, idx[i]]
                np.save(savedir+name+'/'+str(i)+'.npy', tmp)
                
def build_samplesetjpg(namelist, datapath, ssize, savedir):
    if not os.path.isdir(savedir):
        os.mkdir(savedir)
    for name in namelist:
        if not os.path.isdir(savedir+name):
            os.mkdir(savedir+name)
        group = name.split('_')[0]
        with h5py.File(datapath+group+'/'+name, 'r') as f:
            print(name)
            if f['bag'].shape[0]<ssize:
                data_size=f['bag'].shape[0]
                idx = np.arange(data_size)
#                 idx=np.random.choice(f['bag'].shape[0],size=f['bag'].shape[0],replace=False)
            else:
                data_size=ssize
                idx=np.random.choice(f['bag'].shape[0],size=data_size,replace=False)
            label = f['label'][()]
            print(label)
            for i in range(data_size):
                if i%100 == 0:
                    print('{}/{}'.format(i+1, data_size))
                img = f['bag'][idx[i]]
                img = np.transpose(img, (1, 2, 0))
                matplotlib.image.imsave(savedir+name+'/'+str(i)+'.jpg', img)


def run():
    tumorslide = ['tumor_009.h5', 'tumor_011.h5', 'tumor_016.h5', 'tumor_046.h5', 'tumor_054.h5', 
                  'tumor_076.h5', 'tumor_078.h5', 'tumor_085.h5', 'tumor_089.h5', 'tumor_090.h5', 
                  'tumor_095.h5', 'tumor_110.h5']

    # build training set for discriminator
    namelist = tumorslide + random.choices(os.listdir(normaldir), k=20)
    build_sampleset(goodtestslide, './datasets/camelyon_1level224/', 2000, './datasets/camelyon_1level224/selected_sample_testing/')

    
    bsize=104

    normalnamelist = glob.glob('./datasets/camelyon_1level224/selected_sample_training_norm/normal*/*')
    tumornamelist = glob.glob('./datasets/camelyon_1level224/selected_sample_training_norm/tumor*/*')

    trainset0, valset0 =  train_test_split(normalnamelist, test_size=0.2, random_state=42)
    trainset1, valset1 =  train_test_split(tumornamelist, test_size=0.2, random_state=42)

    # trainset = camelyon_data.trainingname_batch_split(trainset0+valset0, trainset1+valset1, 7, 6)
    # valset = camelyon_data.trainingname_batch_split(valset0, valset1, 7, 6)

    def trainneggen():
        for name in trainset0+valset0:
            tmp = np.load(name, allow_pickle=True).item()
            yield tmp['patch'], tmp['label']
            
    def trainposgen():
        for name in trainset1+valset1:
            tmp = np.load(name, allow_pickle=True).item()
            yield tmp['patch'], tmp['label']

    def valgen():
        for name in valset:
            tmp = np.load(name, allow_pickle=True).item()
            yield tmp['patch'], tmp['label']
        

    ds_train=[tf.data.Dataset.from_generator(generator=trainneggen, output_types=(tf.int8, tf.int32),\
                                                output_shapes=(tf.TensorShape([3, 224, 224]),tf.TensorShape([])))\
            .map(load_discimage, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(23838),
            tf.data.Dataset.from_generator(generator=trainposgen, output_types=(tf.int8, tf.int32),\
                                                output_shapes=(tf.TensorShape([3, 224, 224]),tf.TensorShape([])))\
            .map(load_discimage, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(23999)]

    ds_train = tf.data.experimental.sample_from_datasets(ds_train, weights=[0.5, 0.5]).batch(bsize)\
                .prefetch(tf.data.experimental.AUTOTUNE)


    ds_val=tf.data.Dataset.from_generator(generator=valgen, output_types=(tf.int8, tf.int32),\
                                                output_shapes=(tf.TensorShape([3, 224, 224]),tf.TensorShape([])))\
            .map(load_discimage, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
            .batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)

    print('pipeline built')


    # load model
    tf.keras.backend.clear_session()

    resnet50 = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet',input_shape=(224, 224, 3))
    resnet_block3 = models.Model(inputs = resnet50.input, 
                                outputs = resnet50.get_layer('conv4_block6_out').output, name='resnet_block3')

    disc_model = models.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(224, 224, 3),name='enrescale'),
    resnet_block3,
    layers.GlobalAveragePooling2D(name='globalpooling'),
    layers.Dropout(rate=0.25,name='endrop'),
    layers.Dense(1, activation='sigmoid', name='enhead'),

    ],name='discriminator')
    disc_model.summary()


    STEP_PER_EPOCH = len(trainset)//bsize
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.0002,
    decay_steps=STEP_PER_EPOCH*150,
    decay_rate=1,
    staircase=False)

    checkpoint_path = './models/camelyondisc_1level224_checkpoints/discriminator/'+"cp-{epoch:04d}/"
    def get_callbacks():
    return [tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_best_only=True,
                                                save_weights_only=False, monitor='val_accuracy', save_freq='epoch')]

    # weight_for_0 = 0.93 #(1 / 14)*26/2.0 0.93
    # weight_for_1 = 1.083 #(1 / 12)*26/2.0 1.083

    # class_weight = {0: weight_for_0, 1: weight_for_1}

    disc_model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy', tf.keras.metrics.AUC(), sensitivity, specificity])
    history = disc_model.fit(
    ds_train,
    validation_data=ds_val,
    epochs=150,
    callbacks=get_callbacks()
    )

    # intelligent sampling
    intelsampling.intelsampling(datanamelist, 224, 1024, disc_model, encoder, embedding_dir, 'resnet', sampling=True)


if __name__ == '__main__':
    run()