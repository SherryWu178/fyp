import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint, Callback
import tensorflow.keras.backend as K

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

history_path = "history.csv"

train_image = np.load('../urban_input_train.npz')['arr_0']
val_image = np.load('../urban_input_valid.npz')['arr_0']
test_image = np.load('../urban_input_test.npz')['arr_0']

component = 'w'

if component == 'u':

  train_labels = np.load('../urban_u_train.npz')['arr_0']
  val_labels = np.load('../urban_u_valid.npz')['arr_0']
  test_labels = np.load('../urban_u_test.npz')['arr_0']    

elif component == 'v':

  train_labels = np.load('../urban_v_train.npz')['arr_0']
  val_labels = np.load('../urban_v_valid.npz')['arr_0']
  test_labels = np.load('../urban_v_test.npz')['arr_0']    

elif component == 'w':

  train_labels = np.load('../urban_w_train.npz')['arr_0']
  val_labels = np.load('../urban_w_valid.npz')['arr_0']
  test_labels = np.load('../urban_w_test.npz')['arr_0']    


def create_nn(lw = 0, loss_fn = 'mae'):
    # input layers (512 x 512 x 1)
    inputs = layers.Input((256, 256, 1))

    # initializer
    k_init = 'glorot_uniform'

    # regularizer
    k_reg = tf.keras.regularizers.l1(l=lw)  # 1e-5 ?

    # kernal size
    ks = 5

    # filter number
    n_filter = 32

    # first layer
    c01 = layers.Conv2D(filters=n_filter, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(inputs)
    c01 = layers.BatchNormalization()(c01)
    c01 = layers.Activation('relu')(c01)
    c01 = layers.Conv2D(filters=n_filter, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(c01)
    c01 = layers.BatchNormalization()(c01)
    c01 = layers.Activation('relu')(c01)  
#     print(f"{c01.shape = }")

    # 512x512x8 -> 256x256x8
    m01 = layers.MaxPooling2D(pool_size=2)(c01)
#     print(f"{m01.shape = }")

    c02 = layers.Conv2D(filters=n_filter*2, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(m01)
    c02 = layers.BatchNormalization()(c02)
    c02 = layers.Activation('relu')(c02)
    c02 = layers.Conv2D(filters=n_filter*2, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(c02)
    c02 = layers.BatchNormalization()(c02)
    c02 = layers.Activation('relu')(c02)   
#     print(f"{c02.shape = }")

    # 256x256x8 -> 128x128x8
    m02 = layers.MaxPooling2D(pool_size=2)(c02)
#     print(f"{m02.shape = }")

    c03 = layers.Conv2D(filters=n_filter*4, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(m02)
    c03 = layers.BatchNormalization()(c03)
    c03 = layers.Activation('relu')(c03)
    c03 = layers.Conv2D(filters=n_filter*4, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(c03)
    c03 = layers.BatchNormalization()(c03)
    c03 = layers.Activation('relu')(c03)   
#     print(f"{c03.shape = }")

    # 128x128x8 -> 64x64x8
    m03 = layers.MaxPooling2D(pool_size=2)(c03)
#     print(f"{m03.shape = }")

    # bottom of |_|
    c04 = layers.Conv2D(filters=n_filter*8, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(m03)
    c04 = layers.BatchNormalization()(c04)
    c04 = layers.Activation('relu')(c04)
    c04 = layers.Conv2D(filters=n_filter*8, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(c04)
    c04 = layers.BatchNormalization()(c04)
    c04 = layers.Activation('relu')(c04)  
#     print(f"{c04.shape = }")

    # 64x64x8 -> 32x32x8
    m04 = layers.MaxPooling2D(pool_size=2)(c04)
#     print(f"{m04.shape = }")

    # bottom of |_|
    c05 = layers.Conv2D(filters=n_filter*16, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(m04)
    c05 = layers.BatchNormalization()(c05)
    c05 = layers.Activation('relu')(c05)
    c05 = layers.Conv2D(filters=n_filter*16, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(c05)
    c05 = layers.BatchNormalization()(c05)
    c05 = layers.Activation('relu')(c05)
#     print(f"{c05.shape = }")

    # 32x32x8 -> 64x64x8 -> 64x64x[8+8]
    u04 = layers.UpSampling2D(size=2)(c05)
    u04 = layers.concatenate([u04, c04])
#     print(f"{u04.shape = }")

    c06 = layers.Conv2D(filters=n_filter*8, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(u04)
    c06 = layers.BatchNormalization()(c06)
    c06 = layers.Activation('relu')(c06)
    c06 = layers.Conv2D(filters=n_filter*8, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(c06)
    c06 = layers.BatchNormalization()(c06)
    c06 = layers.Activation('relu')(c06)  
#     print(f"{c06.shape = }")


    # 64x64x8 -> 128x128x8 -> 128x128x[8+8]
    u03 = layers.UpSampling2D(size=2)(c06)
    u03 = layers.concatenate([u03, c03])
#     print(f"{u03.shape = }")

    c07 = layers.Conv2D(filters=n_filter*4, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(u03)
    c07 = layers.BatchNormalization()(c07)
    c07 = layers.Activation('relu')(c07)
    c07 = layers.Conv2D(filters=n_filter*4, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(c07)
    c07 = layers.BatchNormalization()(c07)
    c07 = layers.Activation('relu')(c07)  
#     print(f"{c07.shape = }")


    # 128x128x8 -> 256x256x8 -> 256x256x[8+8]
    u02 = layers.UpSampling2D(size=2)(c07)
    u02 = layers.concatenate([u02, c02])
#     print(f"{u02.shape = }")

    c08 = layers.Conv2D(filters=n_filter*2, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(u02)
    c08 = layers.BatchNormalization()(c08)
    c08 = layers.Activation('relu')(c08)
    c08 = layers.Conv2D(filters=n_filter*2, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(c08)
    c08 = layers.BatchNormalization()(c08)
    c08 = layers.Activation('relu')(c08)  
#     print(f"{c08.shape = }")


    # 256x128x8 -> 512x512x8 -> 512x512x[8+8]
    u01 = layers.UpSampling2D(size=2)(c08)
    u01 = layers.concatenate([u01, c01])
#     print(f"{u01.shape = }")

    c09 = layers.Conv2D(filters=n_filter, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(u01)
    c09 = layers.BatchNormalization()(c09)
    c09 = layers.Activation('relu')(c09)
    c09 = layers.Conv2D(filters=n_filter, kernel_size=ks, padding='same', kernel_regularizer=k_reg, kernel_initializer=k_init)(c09)
    c09 = layers.BatchNormalization()(c09)
    c09 = layers.Activation('relu')(c09)  
#     print(f"{c09.shape = }")

    u00 = layers.Conv2D(filters=1, kernel_size=1, padding='same', kernel_initializer=k_init)(c09)
#     print(f"{u00.shape = }")

    # initiate model
    nn = models.Model(inputs=inputs, outputs=u00)  

    # optimizer
    optimizer = tf.keras.optimizers.Adam(0.001)

    # compile model with [?] loss
    nn.compile(loss = loss_fn, optimizer = optimizer, metrics = ['mse', 'mae'])
    
    return nn

# training setting
LOSS_FN = 'mae'
LW = 1e-9 #0

# initiate NN model
nn = create_nn(lw = LW, loss_fn = LOSS_FN)
nn.summary()

EPOCH = 800
BS = 8
LR = 1e-3

# callback setting: learning rate schedule
lr_sched = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=5e-6)
csv_logger = keras.callbacks.CSVLogger(history_path, append=True)
callbacks_list = [lr_sched, csv_logger]

# time it
t0 = time.time()

# train NN model
K.set_value(nn.optimizer.lr, LR)  # set learning rate
history = nn.fit(x=train_image, y=train_labels, validation_data=(val_image, val_labels),
                 batch_size=BS, epochs=EPOCH, verbose=2, callbacks=callbacks_list)

# print ("...\nRunning time: %d mins %d secs!" %(int(time()-t0)/60, np.remainder(int(time()-t0), 60)))

predicted = nn.predict(val_image,verbose=0)
err = predicted - val_labels
print(err)

if component == 'u':
  model_name =  "u_vel_model.h5py"
elif component =='v':
  model_name =  "v_vel_model.h5py"
elif component =='w':
  model_name =  "w_vel_model.h5py"
  
nn.save(model_name)

predicted = nn.predict(test_image,verbose=0)
err = np.mean(np.abs(predicted - test_labels))
print(err)

for i in range(3):

  n_case = i*4 + 2

  plt.subplot(3,3,3*i+1)
  plt.imshow(val_labels[n_case,:,:,0], vmin=-0.6,vmax=0.6)
  plt.colorbar()
  plt.title("Ref")
  plt.subplot(3,3,3*i+2)
  plt.imshow(predicted[n_case,:,:,0], vmin=-0.6,vmax=0.6)
  plt.colorbar()
  plt.title("Predicted")
  plt.subplot(3,3,3*i+3)
  plt.imshow(err[n_case,:,:,0], vmin=-0.1,vmax=0.1)
  plt.colorbar()
  plt.title("Err")

plt.savefig("result.png")
