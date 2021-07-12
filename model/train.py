import os
import matplotlib.pyplot as plt
import numpy as np
base_dir = 'E:/Python/TensorFlow/tensorflow/cancer2995/lung_colon_image_set'
mfvep_dir = os.path.join(base_dir, 'colon_image_sets')
mfvep_normal_dir = os.path.join(mfvep_dir, 'colon_n')
mfvep_abnormal_dir = os.path.join(mfvep_dir, 'colon_aca')

print('total training normal images:', len(os.listdir(mfvep_normal_dir)))
print('total training abnormal images:', len(os.listdir(mfvep_abnormal_dir)))

from PIL import Image
#print(np.array(Image.open('E:/cancer2995/lung_colon_image_set/colon_image_sets/colon_n/colonn1.jpeg')).shape)



from tensorflow.keras import layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (125, 125, 3)))

model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(64, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Conv2D(128, (3, 3), activation = 'relu'))

model.add(layers.MaxPooling2D(2, 2))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dense(1, activation = 'sigmoid'))

model.summary()

train_dir = os.path.join(base_dir, 'train')
train_normal_dir = os.path.join(train_dir, 'colon_n')
train_abnormal_dir = os.path.join(train_dir, 'colon_aca')
validation_dir = os.path.join(base_dir, 'validation')
validation_normal_dir = os.path.join(validation_dir, 'normal')
validation_abnormal_dir = os.path.join(validation_dir, 'abnormal')

#print(validation_dir)

import shutil
import random
# for x in range(500):
#     files = [file for file in os.listdir(train_normal_dir) if os.path.isfile(os.path.join(train_normal_dir, file))]
#     file = random.choice(files)
#
#     shutil.move(os.path.join(train_normal_dir, file), validation_normal_dir)
# for x in range(500):
#     files = [file for file in os.listdir(train_abnormal_dir) if os.path.isfile(os.path.join(train_abnormal_dir, file))]
#     file = random.choice(files)
#
#     shutil.move(os.path.join(train_abnormal_dir, file), validation_abnormal_dir)
#model.save('model.h5')
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# train_datagen = ImageDataGenerator(rescale = 1./255)
# test_datagen = ImageDataGenerator(rescale = 1./255)

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (125, 125),
    batch_size = 20,
    class_mode = 'binary'
    )

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size = (125, 125),
    batch_size = 20,
    class_mode = 'binary'
    )

from tensorflow.keras import optimizers
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizers.RMSprop(lr = 1e-4),
              metrics = ['acc']
              )
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 500,
    epochs = 50,
    validation_data = validation_generator,
    validation_steps = 50
)

model.save('colon_image_sets.h5')

print(history.history.keys())


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure(0)
plt.plot(epochs, acc, 'b-*', label = 'Training acc')
plt.plot(epochs, val_acc, 'r-+', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure(1)
plt.plot(epochs, loss, 'b-*', label = 'Training loss')
plt.plot(epochs, val_loss, 'r-+', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()