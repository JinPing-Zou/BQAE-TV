
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

def quaternion_init(in_features, out_features, rng, init_criterion):
    fan_in, _ = in_features, out_features
    scale = 2.0 / fan_in if init_criterion == 'he' else 1.0 / fan_in
    kernel_shape = (in_features, out_features)
    number_of_weights = np.prod(kernel_shape)
    
    v_r = rng.normal(0.0, scale, number_of_weights).reshape(kernel_shape)
    v_i = rng.normal(0.0, scale, number_of_weights).reshape(kernel_shape)
    v_j = rng.normal(0.0, scale, number_of_weights).reshape(kernel_shape)
    v_k = rng.normal(0.0, scale, number_of_weights).reshape(kernel_shape)
    
    return v_r, v_i, v_j, v_k



class QuaternionLinear(tf.keras.layers.Layer):
    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='he', weight_init='quaternion',
                 seed=None, **kwargs):
        super(QuaternionLinear, self).__init__(**kwargs)
        self.in_features = in_features // 4
        self.out_features = out_features // 4
        self.r_weight = None
        self.i_weight = None
        self.j_weight = None
        self.k_weight = None

        if bias:
            self.bias = self.add_weight(shape=(self.out_features * 4,), initializer='zeros', trainable=True)
        else:
            self.bias = None

        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0, 1234)
        self.rng = np.random.RandomState(self.seed)

    def build(self, input_shape):
        r_w, i_w, j_w, k_w = quaternion_init(self.in_features, self.out_features, self.rng, self.init_criterion)
        self.r_weight = self.add_weight(shape=(self.in_features, self.out_features), initializer=tf.constant_initializer(r_w), trainable=True)
        print(f"self.r_weight shape: {self.r_weight.shape}")
        self.i_weight = self.add_weight(shape=(self.in_features, self.out_features), initializer=tf.constant_initializer(i_w), trainable=True)
        self.j_weight = self.add_weight(shape=(self.in_features, self.out_features), initializer=tf.constant_initializer(j_w), trainable=True)
        self.k_weight = self.add_weight(shape=(self.in_features, self.out_features), initializer=tf.constant_initializer(k_w), trainable=True)

    def call(self, input):
        if len(input.shape) == 3:
            T, N, C = input.shape
            input = tf.reshape(input, [T * N, C])
            output = self.quaternion_linear(input)
            output = tf.reshape(output, [T, N, output.shape[1]])
        elif len(input.shape) == 2:
            output = self.quaternion_linear(input)
        else:
            raise NotImplementedError
        return output

    def quaternion_linear(self, input):
        cat_kernels_4_r = tf.concat([self.r_weight, -self.i_weight, -self.j_weight, -self.k_weight], axis=0)
        print(f"cat_kernels_4_r shape: {cat_kernels_4_r.shape}")
        cat_kernels_4_i = tf.concat([self.i_weight, self.r_weight, -self.k_weight, self.j_weight], axis=0)
        cat_kernels_4_j = tf.concat([self.j_weight, self.k_weight, self.r_weight, -self.i_weight], axis=0)
        cat_kernels_4_k = tf.concat([self.k_weight, -self.j_weight, self.i_weight, self.r_weight], axis=0)

        cat_kernels_4_quaternion = tf.concat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], axis=1)
        print(f"cat_kernels_4_quaternion shape: {cat_kernels_4_quaternion.shape}")

        output = tf.matmul(input, cat_kernels_4_quaternion)
        print(f"output shape: {output.shape}")
        
        if self.bias is not None:
            output = output + self.bias
        return output

# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical

# # 加载 CIFAR-10 数据集
# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# import tensorflow as tf
# from tensorflow.keras import layers, Model, Input

# def create_custom_model():
#     inputs = Input(shape=(32, 32, 3))
#     x = layers.Flatten()(inputs)
#     x = layers.Dense(4)(x)
#     x = QuaternionLinear(4, 512)(x)
#     x = layers.Activation('relu')(x)

#     x = QuaternionLinear(512, 256)(x)
#     x = layers.Activation('relu')(x)

#     x = QuaternionLinear(256, 10)(x)
#     outputs = layers.Dense(10)(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# model = create_custom_model()
# model.summary()


# # 编译模型
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])

# # 训练模型
# model.fit(x_train, y_train, epochs=100, batch_size=64, validation_data=(x_test, y_test))

# # 评估模型
# test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
# print(f'\nTest accuracy: {test_acc}')






