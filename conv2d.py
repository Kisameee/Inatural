from keras.metrics import *
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height = 128, 128

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 435130
nb_validation_samples = 24426
epochs = 50
batch_size = 64

# changez le nom à chaque fois svp ↓
experiment_name = "INATURALIST_E500_D512_C16.3.3_Lr0.01_Relu"
tb_callback = TensorBoard("./logs/" + experiment_name, )

print("Model training will start soon")

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(8142))
model.add(Activation('softmax'))

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])


# model.compile(sgd, mse, metrics=['categorical_accuracy'])
model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,

    target_size=(img_width, img_height),
    batch_size=batch_size, shuffle=True, seed=None,
    class_mode='categorical',
    interpolation='nearest')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, shuffle=True, seed=None,
    class_mode='categorical',
    interpolation='nearest')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[tb_callback],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')
