
import tensorflow as tf


def add_noise(model, param_noise_sigma, learning_rate):
    for param in model.trainable_variables:
        if len(param.shape) >= 2:  
            noise = tf.random.normal(param.shape) * param_noise_sigma * learning_rate
            param.assign(param + noise)  



def add_l2_loss(loss, model, weight_decay):
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in model.trainable_variables])
    total_loss = loss + weight_decay * l2_loss
    return total_loss


def total_variation_loss(y_pred):
    height, width = tf.shape(y_pred)[1], tf.shape(y_pred)[2]
    
    x_diff = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]
    y_diff = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
    
    x_diff_last = y_pred[:, :, 0, :] - y_pred[:, :, -1, :]  
    y_diff_last = y_pred[:, 0, :, :] - y_pred[:, -1, :, :]  
    
    x_diff = tf.concat([x_diff, x_diff_last[:, :, tf.newaxis, :]], axis=2)
    y_diff = tf.concat([y_diff, y_diff_last[:, tf.newaxis, :, :]], axis=1)
    
    tv_loss = tf.reduce_mean(tf.sqrt(tf.square(x_diff) + tf.square(y_diff) + 1e-10))
    
    return tv_loss
