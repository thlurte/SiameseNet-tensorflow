
import os
import uuid
import numpy as np

import tensorflow as tf
import tensorflow.nn as tfn
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import tensorflow.keras.layers as tfl
import tensorflow.keras.regularizers as tkr

class SiameseNet():
  def __init__(self):

    self.img_shape=(105,105,3)
    self.batch_size=128

    self.epochs=50
    self.weights_dir=os.getcwd()+"/"+str(uuid.uuid4())+"/"

            #-------------------#
            #  Hyperparameters  #
            #-------------------#

    self.filters={
        "L1":64
        ,"L2":128
        ,"L3":128
        ,"L4":256
    }

    self.kernel_size={
        "L1":(10,10)
        ,"L2":(7,7)
        ,"L3":(4,4)
        ,"L4":(4,4)
    }

    self.l2_regularizer={
        "L1":2e-4
        ,"L2":2e-4
        ,"L3":2e-4
        ,"L4":2e-4
        ,"L5":2e-3
    }

    self.pool_size={
        "L1":2
        ,"L2":2
        ,"L3":2
    }

            #----------------------#
            #  Model Architecture  #
            #----------------------#

    # Left Side
    self.left_input,self.left_output=self._load_architecture()
    # Right Side
    self.right_input,self.right_output=self._load_architecture()

    L1_layer=tfl.Lambda(lambda tensors: K.abs(tensors[0]-tensors[1]))
    L1_siamese_dist = L1_layer([self.left_output, self.right_output])
    L1_siamese_dist = tfl.Dropout(0.4)(L1_siamese_dist)

    # An output layer with Sigmoid activation function
    prediction = tfl.Dense(1, activation='sigmoid',)(L1_siamese_dist)

    siamese_net = tfk.Model(inputs=[self.left_input, self.right_input], outputs=prediction)
    self.siamese_net = siamese_net
    self.siamese_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=[tfk.metrics.CategoricalAccuracy(name='categorical_accuracy')])

  def initialize_weights(self,shape,dtype=None):
    return K.random_normal(shape,mean=0.0,stddev=0.01,dtype=dtype)

  def initialize_bias(self,shape,dtype=None):
    return K.random_normal(shape,mean=0.5,stddev=0.01,dtype=None)

  def _load_architecture(self):

            #-------------------#
            #    Input Layer    #
            # Shape (1,105,105) #
            #-------------------#

    input=tfl.Input(shape=self.img_shape)


            #---------------------#
            # Convolutional Layer #
            #   Shape (1,96,96)   #
            #---------------------#

    x=tfl.Conv2D(
        filters=self.filters["L1"]

        ,kernel_size=self.kernel_size["L1"]

        ,kernel_initializer=self.initialize_weights

        ,kernel_regularizer=tkr.l2(self.l2_regularizer["L1"])

        ,bias_initializer=self.initialize_bias

    )(input)

            #---------------------#
            #   Activation Layer  #
            #---------------------#

    x=tfl.Activation('relu')(x)

            #---------------------#
            #  Max Pooling Layer  #
            #   Shape (1,48,48)   #
            #---------------------#

    x=tfl.MaxPool2D(
        pool_size=self.pool_size["L1"]
    )(x)



            #---------------------#
            # Convolutional Layer #
            #    Shape (1,42,42)  #
            #---------------------#

    x=tfl.Conv2D(
        filters=self.filters["L2"]

        ,kernel_size=self.kernel_size['L2']

        ,kernel_regularizer=tkr.l2(self.l2_regularizer["L2"])

        ,kernel_initializer=self.initialize_weights

        ,bias_initializer=self.initialize_bias

    )(x)

            #---------------------#
            #   Activation Layer  #
            #---------------------#

    x=tfl.Activation('relu')(x)


            #---------------------#
            #  Max Pooling Layer  #
            #   Shape (1,21,21)   #
            #---------------------#

    x=tfl.MaxPool2D(
        pool_size=self.pool_size["L2"]
    )(x)

            #---------------------#
            # Convolutional Layer #
            #   Shape (1,18,18)   #
            #---------------------#

    x=tfl.Conv2D(
        filters=self.filters["L3"]

        ,kernel_size=self.kernel_size["L3"]

        ,kernel_regularizer=tkr.l2(self.l2_regularizer["L3"])

        ,kernel_initializer=self.initialize_weights

        ,bias_initializer=self.initialize_bias
    )(x)

            #---------------------#
            #   Activation Layer  #
            #---------------------#

    x=tfl.Activation('relu')(x)

            #---------------------#
            #  Max Pooling Layer  #
            #    Shape (1,9,9)    #
            #---------------------#

    x=tfl.MaxPool2D(
        pool_size=self.pool_size["L3"]
    )(x)


            #-----------------------#
            # Convolutational Layer #
            #     Shape (1,6,6)     #
            #-----------------------#

    x=tfl.Conv2D(
        filters=self.filters["L4"]

        ,kernel_size=self.kernel_size['L4']

        ,kernel_regularizer=tkr.l2(self.l2_regularizer['L4'])

        ,kernel_initializer=self.initialize_weights

        ,bias_initializer=self.initialize_bias
    )(x)

            #-----------------------#
            #        Flatten        #
            #-----------------------#

    x=tfl.Flatten()(x)

            #-----------------------#
            # Fully Connected Layer #
            #-----------------------#

    x=tfl.Dense(
        units=4096

        ,kernel_regularizer=tkr.l2(self.l2_regularizer['L5'])

        ,kernel_initializer=self.initialize_weights

        ,bias_initializer=self.initialize_bias

        ,activation='sigmoid'
    )(x)

    return input,x


  def summary(self):
      self.siamese_net.summary()


  def loader(self,data_dir):

    x_train_0=tfk.utils.image_dataset_from_directory(
        directory=data_dir+"/0/"
        ,validation_split=0.2
        ,subset='training'
        ,seed=1218
        ,image_size=(self.img_shape[0],self.img_shape[1])
        ,batch_size=self.batch_size
    )

    x_val_0=tfk.utils.image_dataset_from_directory(
        directory=data_dir+"/0/"
        ,validation_split=0.2
        ,subset='validation'
        ,seed=1218
        ,image_size=(self.img_shape[0],self.img_shape[1])
        ,batch_size=self.batch_size
    )

    x_train_1=tfk.utils.image_dataset_from_directory(
        directory=data_dir+"/1/"
        ,validation_split=0.2
        ,subset='training'
        ,seed=1218
        ,image_size=(self.img_shape[0],self.img_shape[1])
        ,batch_size=self.batch_size
    )

    x_val_1=tfk.utils.image_dataset_from_directory(
        directory=data_dir+"/1/"
        ,validation_split=0.2
        ,subset='validation'
        ,seed=1218
        ,image_size=(self.img_shape[0],self.img_shape[1])
        ,batch_size=self.batch_size
    )

    def convert_dataset_to_numpy(dataset):
      data = []
      labels = []
      for batch in dataset.as_numpy_iterator():
          data.append(batch[0])
          labels.append(batch[1])

      return np.vstack(data), np.concatenate(labels)

    self.train_data_0, self.train_labels_0 = convert_dataset_to_numpy(x_train_0)
    self.val_data_0, self.val_labels_0 = convert_dataset_to_numpy(x_val_0)

    self.train_data_1, self.train_labels_1 = convert_dataset_to_numpy(x_train_1)
    self.val_data_1, self.val_labels_1 = convert_dataset_to_numpy(x_val_1)
    print(self.train_data_0.shape,self.train_data_1.shape)

    self.train_labels=[self.train_labels_0, self.train_labels_1]
    self.val_labels=[self.val_labels_0, self.val_labels_1]
    print(self.train_labels_0.shape,self.train_labels_1.shape)


  def fit(self,early_stopping,patience):

    callback=[]

    if early_stopping:
      es=tfk.callbacks.EarlyStopping(monitor='val_loss',patience=patience,mode='auto',verbose=1)
      callback.append(es)

    self.history = self.siamese_net.fit(
        [self.train_data_0,self.train_data_1]
        ,self.train_labels
        ,batch_size=self.batch_size
        ,epochs=self.epochs
        ,validation_data=([self.val_data_0,self.val_data_1],self.val_labels)
        ,callbacks=callback
        ,verbose=1)

    self.siamese_net.save_weights(self.weights_dir)

