import numpy as np
from matplotlib import pyplot as plt
# First, import TF and get its version.
import tensorflow as tf
from tensorflow import keras
tf_version = tf.__version__

# Check if version >=2.0.0 is used
if not tf_version.startswith('2.'):
    print('WARNING: TensorFlow >= 2.0.0 will be used in this course.\nYour version is {}'.format(tf_version) + '.\033[0m')
else:
    print('OK: TensorFlow >= 2.0.0' + '.\033[0m')

# Check if GPU available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load data & split data between train and test sets
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()


# No reshaping but still doing some conversion

X_train =  X_train.astype('float32')    # change the type towards float32
X_test =   X_test.astype('float32')     # item
X_train /= 255                          # normalize the range to be between 0.0 and 1.0
X_test /=  255                          # item                     
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

n_classes = 10
Y_train = keras.utils.to_categorical(y_train, n_classes)    # modify targets to 1-hot using utils.to_categorical()
Y_test = keras.utils.to_categorical(y_test, n_classes)      # idem 
print(Y_train[:10])

E = 30                 # number of epochs
B = 128               # batch size

# ... define the model as a Sequential type

###############################################
model = keras.models.Sequential()

model.add(keras.layers.Conv2D(32, padding='same',
                            activation='ReLU',
                            kernel_size=(3,3),
                            input_shape=(32,32,3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

# layer 2

model.add(keras.layers.Conv2D(64, padding='same',
                            activation='ReLU',
                            kernel_size=(5,5),
                            input_shape=(32,32,3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

# layer 3

model.add(keras.layers.Conv2D(128, padding='same',
                            activation='ReLU',
                            kernel_size=(7,7),
                            input_shape=(32,32,3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))



# ... print model infomration with summary() method
print(model.summary())
###############################################

model.compile(optimizer='adam', 
            loss='categorical_crossentropy',
            metrics=['accuracy'])

log = model.fit(X_train, Y_train, batch_size=B, epochs=E, verbose=1, validation_data=(X_test, Y_test))

f = plt.figure(figsize=(12,4))
ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)
ax1.plot(log.history['loss'], label='Training loss')
ax1.plot(log.history['val_loss'], label='Testing loss')
ax1.legend()
ax1.grid()
ax2.plot(log.history['accuracy'], label='Training acc')
ax2.plot(log.history['val_accuracy'], label='Testing acc')
ax2.legend()
ax2.grid()
plt.show()