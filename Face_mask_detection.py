import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
#print('Success')

main_dir = 'Masks Dataset'
train_dir = os.path.join(main_dir, 'Train')
test_dir = os.path.join(main_dir, 'Test')
valid_dir = os.path.join(main_dir, 'Validation')

train_mask_dir = os.path.join(train_dir, 'Mask')
train_nomask_dir = os.path.join(train_dir, 'Non Mask')

'''
print(os.listdir(train_dir))
print(len(os.listdir(train_mask_dir)))
print(len(os.listdir(train_nomask_dir)))
'''

train_datagen = ImageDataGenerator(rescale=1/255,
                                   rotation_range=40,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

valid_datagen = ImageDataGenerator(rescale=1/255)

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size= 30,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(150, 150),
    batch_size= 30,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size= 30,
    class_mode='binary'
)

#print(train_generator.class_indices)
#print(train_generator.image_shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='SAME', activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), padding='SAME', activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(Adam(lr=0.001),
                   loss='binary_crossentropy',
                   metrics=['acc'])

model.fit(train_generator,
            epochs = 40,
            validation_data=valid_generator
)


#print(history.history_keys())

'''
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.xlabel('epochs')

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training', 'validation'])
plt.xlabel('epochs')
'''

test_loss, test_acc = model.evaluate(test_generator)
print('Test loss:{} Test Accuracy:{}'.format(test_loss, test_acc))

model.save('maskvsnomask.h5')