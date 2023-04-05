import tensorflow as tf
import numpy as np
import h5py
import os
from tensorflow.keras import datasets,layers,models
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
import openpyxl
import random
import glob
import intelsampling
import camelyon_util as cutil
import milmodels

model_dir = './models/camelyondisc_1level224_checkpoints/discriminator.h5'

datanamelist = glob.glob('./datasets/camelyon_1level224/normal/*.h5') + glob.glob('./datasets/camelyon_1level224/tumor/*.h5')

testpath = './datasets/camelyon_1level224/test/*.h5'
testnamelist = glob.glob(testpath)
embedding_dir = './datasets/embedded_cam1level224/'
encoded_shape = 512
K = 0.3


def run():
    normalnamelist = glob.glob('./datasets/camelyon_1level224/normal/*.h5')
    tumornamelist = glob.glob('./datasets/camelyon_1level224/tumor/*.h5')
    datanamelist = intelsampling.trainingname_batch_split(normalnamelist, tumornamelist, 12, 8)

    labelset=np.load(embedding_dir+'resnet'+'labels.npy',allow_pickle=True).item()
    dataset1=np.load(embedding_dir+'resnet'+'embeddeddata.npy',allow_pickle=True).item()
    scoreset=np.load(embedding_dir+'resnet'+'scores.npy',allow_pickle=True).item()

    trainset, trainlabel = intelsampling.createbags_final(datanamelist, scoreset, dataset1, labelset, K, 1024)

    labelset=np.load(embedding_dir+'resnet'+'labels.npy',allow_pickle=True).item()
    dataset1=np.load(embedding_dir+'resnet'+'embeddeddata.npy',allow_pickle=True).item()
    scoreset=np.load(embedding_dir+'resnet'+'scores.npy',allow_pickle=True).item()

    testset, testlabel = intelsampling.createbags_final(testnamelist, scoreset, dataset1, labelset, K, 1024)

    encoded_shape = 1024
    #trainnamelist, testnamelist = kfold_namelist(label_dic, 4, 4, start) # take 4 for test from each class, end to ending index. 4 classes
    bsize = 1
    STEP_PER_EPOCH = len(trainset) // bsize

    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.0002,
        decay_steps=STEP_PER_EPOCH*10,
        decay_rate=1,
        staircase=False)
    def get_callbacks():
        return tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)

    weight_for_1 = (1 / sum(trainlabel)) * len(trainlabel) / 2.0
    weight_for_0 = (1 / (len(trainlabel) - sum(trainlabel))) * len(trainlabel) /2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}


    def traingen():
        for xy in zip(trainset,trainlabel):
            yield xy
            
    def testgen():
        for xy in zip(testset,testlabel):
            yield xy
    print('sets built')
    tf.keras.backend.clear_session()
    ds_train=tf.data.Dataset.from_generator(generator=traingen, output_types=(tf.float32, tf.int32),\
                                            output_shapes=(tf.TensorShape([None,encoded_shape]),tf.TensorShape([])))\
                    .map(cutil.load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                    .shuffle(len(trainset)).batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)

    ds_test=tf.data.Dataset.from_generator(generator=testgen, output_types=(tf.float32, tf.int32),\
                                            output_shapes=(tf.TensorShape([None,encoded_shape]),tf.TensorShape([])))\
                    .map(cutil.load_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                    .batch(bsize).prefetch(tf.data.experimental.AUTOTUNE)
    print('pipeline built')


    AttMILbinary=milmodels.AttMILbinary()
    AttMILbinary.build(inputshape = (None, None, encoded_shape)) # need to change shape and n_class

    AttMILbinary.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
                loss=tf.keras.losses.binary_crossentropy,
                metrics=['accuracy', tf.keras.metrics.AUC(), cutil.specificity, cutil.sensitivity])
    history = AttMILbinary.fit(ds_train,validation_data=ds_test,epochs=150, callbacks = get_callbacks(), class_weight=class_weight)


    predictions=[]
    gts=[]
    for x,y in ds_test:    
        predictions.extend(np.round(AttMILbinary.predict_on_batch(x)).tolist())
        gts.extend(np.round(y).tolist())
    mat=confusion_matrix(gts,predictions)
    plt.figure(1)
    plot_confusion_matrix(conf_mat=mat)
    plt.show()
    AttMILbinary.evaluate(ds_test)
    print('test - spe: {} - sen: {}'.format(round(mat[0, 0]/sum(mat[0]), 3), round(mat[1, 1]/sum(mat[1]), 3)))

    plt.figure(2)
    cutil.plot_metric(history,'loss')
    plt.figure(3)
    cutil.plot_metric(history,'accuracy')
    plt.figure(4)
    cutil.plot_metric(history,'auc')

if __name__ == '__main__':
    run()