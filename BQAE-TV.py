
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from regular import add_noise, add_l2_loss, total_variation_loss
from Model import autoencoder

from SSIM import ssim
from PSNR import psnr

np.random.seed(5)
tf.random.set_seed(5)

image_path = 'pepper.bmp' 
lena_image = Image.open(image_path)

lena_image = lena_image.resize((256, 256))  
lena_image = np.array(lena_image) / 255.0  
lena_image = np.expand_dims(lena_image, axis=0)

mask = np.ones_like(lena_image)
mask_rate = 0.7
mask_indices = np.random.choice([0, 1], size=mask.shape[1:3], p=[mask_rate, 1-mask_rate])

mask[0, ..., :] = np.expand_dims(mask_indices, axis=-1)
masked_lena_image = mask * lena_image


input_shape = (256, 256, 3)
autoencoder = autoencoder()
autoencoder.summary()

Q = masked_lena_image
Omega = mask

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

learning_rate = 1e-3
param_noise_sigma = 2.0
reg_noise_std = 1.0 / 30.0
weight_decay = 1e-5


@tf.function
def train_step(Z_value, Q, Omega):
    with tf.GradientTape() as tape:
        Z = tf.identity(Z_value)
        Z_input = Z + tf.random.normal(Z.shape) * reg_noise_std
        autoencoder_output = autoencoder(Z_input, training=True)
        P_Omega = tf.cast(Omega, tf.float32) * tf.cast(autoencoder_output, tf.float32)
        P_Omega_Q = tf.cast(Omega, tf.float32) * tf.cast(Q, tf.float32)
        re_loss = tf.reduce_mean(tf.square(P_Omega - P_Omega_Q))
        tv_loss = total_variation_loss(autoencoder_output)
        re_loss = add_l2_loss(re_loss, autoencoder, weight_decay)
        loss =re_loss + 0.001*tv_loss
    gradients = tape.gradient(loss, autoencoder.trainable_variables)
    optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
    return loss

psnr_list = []
ssim_list = []
num_epochs = 10000
for epoch in range(num_epochs):
    Z_value = np.random.rand(1,256,256,3)  
    Z_value = tf.constant(Z_value, dtype=tf.float32)
    loss_value = train_step(Z_value, Q, Omega)
    add_noise(autoencoder, param_noise_sigma=0.1, learning_rate=0.001)
    print(f'Epoch {epoch+1}, Loss: {loss_value.numpy()}')
    if epoch % 500== 0:
        X_opt = autoencoder(Z_value, training=False)
        X_inpainted = masked_lena_image + (1 - Omega) * X_opt
        X_inpainted_np = X_inpainted.numpy().squeeze().astype(np.float64)
        lena_image = lena_image.reshape(256, 256, 3)
        X_inpainted_np = X_inpainted_np.reshape(256, 256, 3)
        psnr_value = psnr(lena_image, X_inpainted_np, 1)
        psnr_list.append(psnr_value)
        ssim_value = ssim(lena_image, X_inpainted_np)
        ssim_list.append(ssim_value)
        print(f"Iteration {epoch}, PSNR: {psnr_value:.2f}")
        
        
X_opt_np = X_opt.numpy().squeeze().astype(np.float64)
X_opt_np = X_opt_np.reshape(256, 256, 3)

Z_value = np.random.rand(1,256,256,3)  
Z_value = tf.constant(Z_value, dtype=tf.float32)

X_opt = autoencoder(Z_value, training=False)
X = np.array((1 - Omega) * X_opt)[0]

X_inpainted = masked_lena_image + (1 - Omega) * X_opt
X_inpainted_np = X_inpainted.numpy().squeeze().astype(np.float64)

lena_image = lena_image.reshape(256, 256, 3)
masked_lena_image = masked_lena_image.reshape(256, 256, 3)
X_inpainted_np = X_inpainted_np.reshape(256, 256, 3)


plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(lena_image)

plt.subplot(1, 3, 2)
plt.title("masked Image")
plt.imshow(masked_lena_image)

plt.subplot(1, 3, 3)
plt.title("Inpainted Image")
plt.imshow(X_inpainted_np) 
plt.show()

psnr_value = psnr(lena_image, X_inpainted_np, 1)
print("PSNR: ", psnr_value)

ssim_value = ssim(lena_image, X_inpainted_np)
print("SSIM: ", ssim_value)

psnr_value_init = psnr(lena_image, masked_lena_image, 1)
print("PSNR_init: ", psnr_value_init)

ssim_value_init = ssim(lena_image, masked_lena_image)
print("SSIM_init: ", ssim_value_init)


