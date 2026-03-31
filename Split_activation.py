import tensorflow as tf

def quaternion_split_activation(quaternion_tensor):
    
    w, x, y, z = tf.split(quaternion_tensor, num_or_size_splits=4, axis=-1)

    # w = tf.nn.relu(w)
    # x = tf.nn.relu(x)
    # y = tf.nn.relu(y)
    # z = tf.nn.relu(z)
    w = tf.nn.leaky_relu(w, alpha=0.2)
    x = tf.nn.leaky_relu(x, alpha=0.2)
    y = tf.nn.leaky_relu(y, alpha=0.2)
    z = tf.nn.leaky_relu(z, alpha=0.2)
    activated_quaternion = tf.concat([w, x, y, z], axis=-1)
    return activated_quaternion

