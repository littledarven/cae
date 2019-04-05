import keras
from keras.layers import Input,Conv2D,MaxPooling2D,UpSampling2D, ReLU, BatchNormalization, Dense, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
import h5py

batch_size = 32
epochs = 5
inChannel = 1
x, y = 236, 156

f = h5py.File('train_resized.h5', 'r')
data_description = list(f.keys())[0]
train_file = np.array(list(f[data_description]))

(x_train) = train_file

x_train = x_train.reshape(-1, x, y, 1)
max_value = float(x_train.max())
input_img = Input(shape = (x, y, inChannel))

def ConvAutoEnc(input):

	#encoder part
    conv_layer1 = Conv2D(128,[3,3], strides=1, activation = 'relu', padding = 'same')(input)
    relu_layer1 = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(conv_layer1)
    lrn_layer1 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(relu_layer1)
    pool_layer1 = MaxPooling2D(pool_size=(2, 2))(lrn_layer1)
    conv_layer2 = Conv2D(64,[3,3], activation = 'relu', padding = 'same')(pool_layer1)
    relu_layer2 = ReLU(max_value=None, negative_slope=0, threshold=0.0)(conv_layer2)
    lrn_layer2 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(relu_layer2)
    pool_layer2 = MaxPooling2D(pool_size=(2, 2))(lrn_layer2)
    conv_layer3 = Conv2D(32,[3,3], activation = 'relu', padding = 'same')(pool_layer2)
    relu_layer3 = ReLU(max_value=None, negative_slope=0, threshold=0.0)(conv_layer3)
    
    #fully-conected part
    dropout_layer1 = Dropout(rate=0.5, noise_shape=None, seed=None)(relu_layer3)
    dense_layer1 = Dense(2016, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dropout_layer1)
    dropout_layer2 = Dropout(rate=0.5, noise_shape=None, seed=None)(dense_layer1)
    dense_layer2 = Dense(504, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dropout_layer2)
    dropout_layer3 = Dropout(rate=0.5, noise_shape=None, seed=None)(dense_layer2)
    dense_layer3 = Dense(168, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dropout_layer3)
    dropout_layer4 = Dropout(rate=0.5, noise_shape=None, seed=None)(dense_layer3)
    dense_layer4 = Dense(355, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dropout_layer4)
	# --
    dense_layer5 = Dense(355, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dense_layer4)
    d_dropout_layer1 = Dropout(rate=0.5, noise_shape=None, seed=None)(dense_layer5)
    dense_layer6 = Dense(168, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dense_layer5)
    d_dropout_layer2 = Dropout(rate=0.5, noise_shape=None, seed=None)(dense_layer6)
    dense_layer7 = Dense(504, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dense_layer6)
    d_dropout_layer3 = Dropout(rate=0.5, noise_shape=None, seed=None)(dense_layer7)
    dense_layer8 = Dense(2016, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(dense_layer7)
    d_dropout_layer4 = Dropout(rate=0.5, noise_shape=None, seed=None)(dense_layer8)
    
    #decoder part
    d_relu_layer1 = ReLU(max_value=None, negative_slope=0, threshold=0.0)(dense_layer1)
    deconv_layer1 = Conv2D(32,[3,3], activation = 'relu', padding = 'same')(d_relu_layer1)
    unpool_layer1 = UpSampling2D((2,2))(deconv_layer1)
    d_lrn_layer1 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(unpool_layer1)
    d_relu_layer2 = ReLU(max_value=None, negative_slope=0, threshold=0.0)(d_lrn_layer1)
    deconv_layer2 = Conv2D(64,[3,3], activation = 'relu', padding = 'same')(d_relu_layer2)
    unpool_layer2 = UpSampling2D((2,2))(deconv_layer2)
    d_lrn_layer2 = BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones')(unpool_layer2)
    d_relu_layer3 = ReLU(max_value=None, negative_slope=0, threshold=0.0)(d_lrn_layer2)
    decoded = Conv2D(1,[3,3], activation = 'sigmoid', padding = 'same')(d_relu_layer3)
	
    return decoded

autoencoder = Model(input_img, ConvAutoEnc(input_img))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.summary()

autoencoder_train = autoencoder.fit(x_train, x_train, batch_size=batch_size,epochs=epochs,verbose=1)

autoencoder.save_weights('weights.h5')



#loss = autoencoder_train.history['loss']
#val_loss = autoencoder_train.history['val_loss']
#epochs = range(epochs)
#plt.figure()
#plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.legend()
#plt.show()
