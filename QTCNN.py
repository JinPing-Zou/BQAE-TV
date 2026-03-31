
import tensorflow as tf
import numpy as np
from numpy.random import RandomState
from tensorflow.keras.initializers import RandomUniform, GlorotUniform

import tensorflow as tf
from tensorflow.keras.layers import Layer


class QuaternionDeconv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding='SAME', groups=1, bias=True,
                  weight_init='quaternion', rotation=False, quaternion_format=True, scale=False, name=None):
        super(QuaternionDeconv, self).__init__(name=name)
        
        self.in_channels = in_channels // 4
        self.out_channels = out_channels // 4
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight_init = weight_init
        self.rotation = rotation
        self.quaternion_format = quaternion_format
        self.scale = scale
        
        # Initialize weights
        self.r_weight = self.add_weight(name='d_r_weight',
                                        shape=(kernel_size, kernel_size, self.out_channels, self.in_channels),
                                        initializer='random_normal',
                                        trainable=True)
        self.i_weight = self.add_weight(name='d_i_weight',
                                        shape=(kernel_size, kernel_size, self.out_channels, self.in_channels),
                                        initializer='random_normal',
                                        trainable=True)
        self.j_weight = self.add_weight(name='d_j_weight',
                                        shape=(kernel_size, kernel_size, self.out_channels, self.in_channels),
                                        initializer='random_normal',
                                        trainable=True)
        self.k_weight = self.add_weight(name='d_k_weight',
                                        shape=(kernel_size, kernel_size, self.out_channels, self.in_channels),
                                        initializer='random_normal',
                                        trainable=True)

        if self.scale:
            self.scale_param = self.add_weight(name='d_scale_param',
                                                shape=self.r_weight.shape, initializer='random_normal', trainable=True)
        else:
            self.scale_param = None

        if self.rotation:
            self.zero_kernel = self.add_weight(name='d_zero_kernel',
                                                shape=self.r_weight.shape, initializer='zeros', trainable=False)
        else:
            self.zero_kernel = None

        if bias:
            self.bias = self.add_weight(name='d_bias',
                                        shape=(out_channels,), initializer='zeros', trainable=True)
        else:
            self.bias = None
        
        
        
        
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        output_shape = (batch_size, height * self.stride, width * self.stride, self.out_channels * 4)
        
        # Concatenate weights for quaternion deconvolution
        # a = tf.zeros_like(self.r_weight)
        # cat_kernels_4_r = tf.concat([a,a,a,a], axis=-1)
        cat_kernels_4_r = tf.concat([self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], axis=-1, name='d_cat_kernels_4_r')
        cat_kernels_4_i = tf.concat([self.i_weight, self.r_weight, self.k_weight, -self.j_weight], axis=-1, name='d_cat_kernels_4_i')
        cat_kernels_4_j = tf.concat([self.j_weight, -self.k_weight, self.r_weight, self.i_weight], axis=-1, name='d_cat_kernels_4_j')
        cat_kernels_4_k = tf.concat([self.k_weight, self.j_weight, -self.i_weight, self.r_weight], axis=-1, name='d_cat_kernels_4_k')
        cat_kernels_4_quaternion = tf.concat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], axis=-2, name='d_cat_kernels_4_quaternion')

        # print(f"output_shape shape: {output_shape.shape}")
        # print(f"r_weight shape: {self.r_weight.shape}")
        # print(f"i_weight shape: {self.i_weight.shape}")
        # print(f"j_weight shape: {self.j_weight.shape}")
        # print(f"k_weight shape: {self.k_weight.shape}")
        # print(f"bias shape: {self.bias.shape}")
        # print(f"cat_kernels_4_r shape: {cat_kernels_4_r.shape}")
        # print(f"cat_kernels_4_quaternion shape: {cat_kernels_4_quaternion.shape}")
        
        # Perform the deconvolution
        output = tf.nn.conv2d_transpose(inputs, cat_kernels_4_quaternion, output_shape=output_shape,
                                        strides=[1, self.stride, self.stride, 1], padding=self.padding, name='d_output_deconv')
        
        if self.bias is not None:
            output = tf.add(output, self.bias, name='d_output_bias')
        
        return output


# import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical
# import numpy as np
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


# # Build the CNN model using QuaternionDeconv
# model = tf.keras.Sequential([
#     tf.keras.layers.InputLayer(input_shape=(32, 32, 4)),  # CIFAR-10 image shape
    
#     tf.keras.layers.Conv2D(128, kernel_size=3, strides=1, padding='SAME'),
#     tf.keras.layers.ReLU(),
    
#     QuaternionDeconv(in_channels=128, out_channels=64, kernel_size=3, stride=2),
#     tf.keras.layers.ReLU(),
#     QuaternionDeconv(in_channels=64, out_channels=16, kernel_size=3, stride=2),
#     tf.keras.layers.ReLU(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
# ])
# # Compile the model

# model.summary()
# model.compile(optimizer='adam',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])

# model.fit(x_train, y_train, batch_size=16, epochs=50, validation_data=(x_test, y_test))



