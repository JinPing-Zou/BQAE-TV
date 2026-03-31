import tensorflow as tf

from tensorflow.keras.layers import Input, Flatten, Dense,BatchNormalization, Activation,Dropout,Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models

from QCNN import QuaternionConv
from QTCNN import QuaternionDeconv
from QFCN import QuaternionLinear
from Split_activation import quaternion_split_activation

def autoencoder():
    x = tf.keras.Input(shape=(256, 256, 3), name='input')
    
    h = layers.Conv2D(4, (3, 3), strides=1, padding='same', activation='relu')(x)

    h = QuaternionConv(in_channels=4, out_channels=16, kernel_size=3, stride=1, name='encoder_1')(h)
    h = quaternion_split_activation(h)

    h = QuaternionConv(in_channels=16, out_channels=32, kernel_size=3, stride=2, name='encoder_2')(h)
    h = quaternion_split_activation(h)
    
    h = QuaternionConv(in_channels=32, out_channels=64, kernel_size=3, stride=2, name='encoder_3')(h)
    h = quaternion_split_activation(h)

    h = QuaternionConv(in_channels=64, out_channels=256, kernel_size=3, stride=2,  name='encoder_4')(h)
    h = quaternion_split_activation(h)

    h = QuaternionConv(in_channels=256, out_channels=256, kernel_size=3, stride=2, name='encoder_5')(h)
    h = quaternion_split_activation(h)

    h = QuaternionConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, name='encoder_6')(h)
    h = quaternion_split_activation(h)
    
    h = Flatten()(h)
    h = Dense(1024, activation='relu', name='encoder_7')(h)
    h = Dense(512, activation='relu', name='encoder_8')(h)
    y = Dense(256, activation='relu',name='decoder_7')(h)
    y = Dense(128, activation='relu',name='decoder_6')(y)
    y = Dense(64*8*8, activation='relu',name='decoder_5')(y)
    y = Reshape((8, 8, 64))(y)

    y = QuaternionDeconv(in_channels=64, out_channels=128, kernel_size=3, stride=2,name='decoder_4')(y)
    y = quaternion_split_activation(y)
    
    y = QuaternionDeconv(in_channels=128, out_channels=256, kernel_size=3, stride=2,name='decoder_3')(y)
    y = quaternion_split_activation(y)

    y = QuaternionDeconv(in_channels=256, out_channels=128, kernel_size=3, stride=2,name='decoder_2')(y)
    y = quaternion_split_activation(y)

    y = QuaternionDeconv(in_channels=128, out_channels=32, kernel_size=3, stride=2,name='decoder_1')(y)    
    y = quaternion_split_activation(y)
    
    y = QuaternionDeconv(in_channels=32, out_channels=4, kernel_size=3, stride=2,name='decoder_0')(y)
    y = quaternion_split_activation(y)
    
    y = layers.Conv2DTranspose(3, (3, 3), strides=1, padding='same', activation='relu')(y)
    
    return Model(inputs=x, outputs=y)
