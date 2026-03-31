import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from tensorflow.keras.initializers import RandomUniform, GlorotUniform

from tensorflow.keras.layers import Layer

import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np



class QuaternionConv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='SAME', groups=1, bias=True,
                  weight_init='quaternion', rotation=False, quaternion_format=True, scale=False, name=None):
        super(QuaternionConv, self).__init__(name=name)
        
        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight_init = weight_init
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.scale = scale
        
        self.r_weight = self.add_weight(shape=(kernel_size, kernel_size,self.in_channels,self.out_channels),
                                        initializer='random_normal',
                                        trainable=True)
        self.i_weight = self.add_weight(shape=(kernel_size, kernel_size,self.in_channels,self.out_channels),
                                        initializer='random_normal',
                                        trainable=True)
        self.j_weight = self.add_weight(shape=(kernel_size, kernel_size,self.in_channels,self.out_channels),
                                        initializer='random_normal',
                                        trainable=True)
        self.k_weight = self.add_weight(shape=(kernel_size, kernel_size,self.in_channels,self.out_channels),
                                        initializer='random_normal',
                                        trainable=True)
        
        # print(f"r_weight shape: {self.r_weight.shape}")
        # print(f"i_weight shape: {self.i_weight.shape}")
        # print(f"j_weight shape: {self.j_weight.shape}")
        # print(f"k_weight shape: {self.k_weight.shape}")

        if self.scale:
            self.scale_param = tf.Variable(tf.random.normal(self.r_weight.shape, seed=self.seed + 4))
        else:
            self.scale_param = None

        if self.rotation:
            self.zero_kernel = tf.Variable(tf.zeros(self.r_weight.shape), trainable=False)
        
        if bias:
            self.bias = self.add_weight(shape=(out_channels,),initializer='zeros',trainable=True)
        else:
            self.bias = None
        
        
    def call(self, inputs):
        
        # a = tf.zeros_like(self.r_weight)
        # cat_kernels_4_r = tf.concat([a,a,a,a], axis=-2)
        cat_kernels_4_r = tf.concat([self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], axis=-2)
        # print(f"cat_kernels_4_r hape: {cat_kernels_4_r.shape}")
        cat_kernels_4_i = tf.concat([self.i_weight, self.r_weight, -self.k_weight, self.j_weight], axis=-2)
        cat_kernels_4_j = tf.concat([self.j_weight, self.k_weight, self.r_weight, -self.i_weight], axis=-2)
        cat_kernels_4_k = tf.concat([self.k_weight, -self.j_weight, self.i_weight, self.r_weight], axis=-2)

        cat_kernels_4_quaternion = tf.concat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], axis=-1)
        # print(f"cat_kernels_4_quaternion shape: {cat_kernels_4_quaternion.shape}")
        
        # Perform the deconvolution
        output = tf.nn.conv2d(inputs, cat_kernels_4_quaternion,strides=[1, self.stride, self.stride, 1], padding=self.padding, name='d_output_deconv')
        
        if self.bias is not None:
            output = tf.add(output, self.bias, name='e_output_bias')
        
        return output


# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical

# # 加载数据集
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
# x_train = x_train.astype('float32') / 255.0
# x_test = x_test.astype('float32') / 255.0

# def convert_to_quaternion_format(data):
#     zero_channel = np.zeros((data.shape[0], data.shape[1], data.shape[2], 1))
#     return np.concatenate([zero_channel, data], axis=-1)

# x_train = convert_to_quaternion_format(x_train)
# x_test = convert_to_quaternion_format(x_test)

# y_train = to_categorical(y_train, 10)
# y_test = to_categorical(y_test, 10)



# from tensorflow.keras.layers import Input, Flatten, Dense,BatchNormalization, Activation,Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.models import Model

# def autoencoder():
#     x = Input(shape=(32, 32, 4), name='input')
    
#     # Encoder
#     h = QuaternionConv(in_channels=4, out_channels=32, kernel_size=3, stride=1)(x)
#     h = BatchNormalization()(h)
#     h = Activation('relu')(h)
#     h = Dropout(0.25)(h)
    
    
#     h = QuaternionConv(in_channels=32, out_channels=64, kernel_size=3, stride=2)(h)
#     h = BatchNormalization()(h)
#     h = Activation('relu')(h)
#     h = Dropout(0.25)(h)
    
#     h = QuaternionConv(in_channels=64, out_channels=128, kernel_size=3, stride=2)(h)
#     h = BatchNormalization()(h)
#     h = Activation('relu')(h)
#     h = Dropout(0.25)(h)
    
#     h = QuaternionConv(in_channels=128, out_channels=256, kernel_size=3, stride=2)(h)
#     h = BatchNormalization()(h)
#     h = Activation('relu')(h)
#     h = Dropout(0.25)(h)
    
#     # h = QuaternionConv(in_channels=256, out_channels=512, kernel_size=3, stride=2)(h)
#     # h = BatchNormalization()(h)
#     # h = Activation('relu')(h)
#     # h = Dropout(0.25)(h)
    
#     # Flatten the encoded feature maps
#     h = Flatten()(h)
#     h = Dense(128, activation='relu')(h)
#     h = Dense(10, activation='softmax')(h)

#     return Model(inputs=x, outputs=h)


# autoencoder_model = autoencoder()
# autoencoder_model.compile(optimizer=Adam(learning_rate=0.001), 
#                           loss='categorical_crossentropy', metrics=['accuracy'])
# autoencoder_model.summary()

# autoencoder_model.fit(x_train, y_train, batch_size=16, epochs=50, validation_data=(x_test, y_test))

# loss, accuracy = autoencoder_model.evaluate(x_test, y_test)
# print(f'Test accuracy: {accuracy}')



