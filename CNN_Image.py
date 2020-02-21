'''
 python version - 3.6.8
 Libraries needed
 1.split_folder
 2.keras
 3.tensorflow
'''


import split_folders
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import nadam
from keras.preprocessing.image import ImageDataGenerator


# Change source and destination
source = "C:/Users/Excalibur/Desktop/Big_data/flowers"
destination = "C:/Users/Excalibur/Desktop/Big_data/dest"


# Split folder used to split data into test,validation and train according to ratio defined
split_folders.ratio(source, output=destination, seed=1337, ratio=(.6, .2, .2))  # default values

# Folder destination of test,val,train
train_folder = "C:/Users/Excalibur/Desktop/Big_data/dest/train"
test_folder = "C:/Users/Excalibur/Desktop/Big_data/dest/test"
val_folder = "C:/Users/Excalibur/Desktop/Big_data/dest/val"


# Using CNN sequential model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Adam optimizer
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest',
    horizontal_flip=True,
    vertical_flip=True)

batch_size = 32

train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = train_datagen.flow_from_directory(
    val_folder,
    target_size=(128, 128),
    batch_size=batch_size,
    class_mode='categorical')

history = model.fit_generator(
    train_generator,
    steps_per_epoch=2600//batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=880//batch_size)


# Code to save the model and weights
model.save("C:/Users/Excalibur/Desktop/Big_data/dest/model_flower1.h5")
model.save_weights("C:/Users/Excalibur/Desktop/Big_data/dest/model_flower_weights1.h5")

'''import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()'''