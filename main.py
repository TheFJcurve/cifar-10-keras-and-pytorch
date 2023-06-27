"""Using keras to tackle the CIFAR 10 dataset, using VGG16 convolutional network, dropouts and data
augmentation. The output model and the graphs are saved in the same folder."""

import tensorflow as tf
from keras import layers
from keras import models
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt


(train_set, train_labels), (test_set, test_labels) = cifar10.load_data()

train_set = train_set / 255.0
test_set = test_set / 255.0

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)

NUM_BATCH_SIZE = 32
NUM_EPOCHS = 20 ## Change this at will. The default value is 100.
NUM_STEPS_PER_EPOCH =  train_set.shape[0] // NUM_BATCH_SIZE 
LEARNING_RATE = 0.002
MOMENTUM = 0.9

conv_base = VGG16(weights="imagenet",
                    include_top=False,
                    input_shape=(32, 32 ,3))


model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(10, activation='sigmoid'))


set_trainable = False

for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:  
        layer.trainable = False


train_datagen = ImageDataGenerator(rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

train_generator = train_datagen.flow(train_set,
                                    train_labels,
                                    batch_size=NUM_BATCH_SIZE)

opt = tf.keras.optimizers.SGD(
    learning_rate=LEARNING_RATE,
    momentum=MOMENTUM,
    name="SGD")

model.compile(loss='categorical_crossentropy',
                optimizer=opt,                
                metrics=['acc'])


history = model.fit(train_generator,
                    steps_per_epoch=NUM_STEPS_PER_EPOCH,
                    epochs=NUM_EPOCHS,
                    validation_data=(test_set, test_labels))

model.save('cifar-10-model-keras.h5')

test_loss, test_acc = model.evaluate(test_set, test_labels, steps=50)
print('test acc:', test_acc)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation acc')
plt.legend()

plt.savefig('accuracy-keras.png')

plt.clf()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()

plt.savefig('loss-keras.png')
