import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import datasets,layers,models
import random


# class AttMIL2(models.Model):
#     def __init__(self):
#         super(AttMIL2, self).__init__()
    
#     def build(self, input_shape, n_class):
#         self.V0 = layers.Dense(input_shape[-1]//2,activation='tanh',use_bias=False)
#         self.U0 = layers.Dense(input_shape[-1]//2,activation='sigmoid',use_bias=False)
#         self.V1 = layers.Dense(input_shape[-1]//2,activation='tanh',use_bias=False)
#         self.U1 = layers.Dense(input_shape[-1]//2,activation='sigmoid',use_bias=False)
#         self.V2 = layers.Dense(input_shape[-1]//2,activation='tanh',use_bias=False)
#         self.U2 = layers.Dense(input_shape[-1]//2,activation='sigmoid',use_bias=False)
        
#         self.Wa0 = layers.Dense(1,use_bias=False)
#         self.Wa1 = layers.Dense(1, use_bias=False)
#         self.Wa2 = layers.Dense(1, use_bias=False)
#         self.softmax = layers.Softmax(axis=1)
#         self.dot = layers.Dot(axes=1)
        
#         self.WC = layers.Dense(3,kernel_regularizer=tf.keras.regularizers.l2(0.00001))
#         self.cat = layers.Concatenate(axis=-1)
        
#         super(AttMIL2,self).build(input_shape)
        
#     def call(self, x):
#         x = x[0]
#         V0 = self.V0(x)
#         U0 = self.U0(x)
#         energy0 = tf.math.multiply(V0,U0)
#         V1 = self.V1(x)
#         U1 = self.U1(x)
#         energy1 = tf.math.multiply(V1,U1)
#         V2 = self.V2(x)
#         U2 = self.U2(x)
#         energy2 = tf.math.multiply(V2,U2)
#         #hs
#         x = tf.expand_dims(x,0)
#         att0 = tf.expand_dims(self.Wa0(energy0),0)
#         att0 = self.softmax(att0)       
#         hs0 = self.dot([att0,x]) # 1,vector_size
        
#         att1 = tf.expand_dims(self.Wa1(energy1),0)
#         att1 = self.softmax(att1)       
#         hs1 = self.dot([att1,x]) # 1,vector_size
        
#         att2 = tf.expand_dims(self.Wa2(energy2),0)
#         att2 = self.softmax(att2)       
#         hs2 = self.dot([att2,x]) # 1,vector_size
               
#         hs = self.cat([hs0,hs1,hs2])
#         hs = tf.squeeze(hs,1)
#         #slide score for classes
#         hs = layers.Dropout(rate=0.1)(hs)
#         s = self.WC(hs)
        
        
#         return s
        
class gatedattention(layers.Layer):
    def __init__(self, channels=64, **kwargs):
        super(gatedattention, self).__init__(**kwargs)
        self.channels = channels
        self.V0 = layers.Dense(channels, activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(1e-5),use_bias=False)
        self.U0 = layers.Dense(channels, activation='sigmoid',kernel_regularizer=tf.keras.regularizers.l2(1e-5),use_bias=False)
        self.Wa0 = layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2(1e-5),use_bias=False)
        self.softmax = layers.Softmax(axis=1)
        self.dot = layers.Dot(axes=1)
    
    def call(self, x):
        x = x[0]
        V0 = self.V0(x)
        U0 = self.U0(x)
        energy0 = tf.math.multiply(V0,U0)

        att0 = tf.expand_dims(self.Wa0(energy0),0)
        att0 = self.softmax(att0)
        x = tf.expand_dims(x,0)
        hs0 = self.dot([att0,x]) # 1,vector_size
        hs = tf.squeeze(hs0,1)
        return att0, hs
        
    def get_config(self):
        config = super(gatedattention, self).get_config()
        config.update({'channels':self.channels})
        return config
        
class AttMILbinary(models.Model):
    def __init__(self):
        super(AttMILbinary, self).__init__()

    def build(self, inputshape):
        self.inputshape = inputshape

        self.gatedattention = gatedattention(inputshape[-1]//2, name='attention')

        self.WC2 = layers.Dense(1,activation='sigmoid')

        super(AttMILbinary,self).build(inputshape)

    def call(self, x):

        att0, hs = self.gatedattention(x)
        hs = layers.Dropout(rate=0.2)(hs)
        s = self.WC2(hs)
        
        return s
        
    def get_config(self):
        config = super(AttMILbinary, self).get_config()
        config.update({'inputshape':self.inputshape})
        return config
