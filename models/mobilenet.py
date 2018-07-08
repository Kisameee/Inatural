from keras import applications
from keras.callbacks import *
from keras.layers import *
from keras.metrics import *
from keras.models import *
from keras.preprocessing.image import ImageDataGenerator

# path to the model weights files.
weights_path = '../keras/examples/vgg16_weights.h5'
# dimensions of our images.

img_width, img_height = 128, 128
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 435130
nb_validation_samples = 24426
epochs = 2
batch_size = 64

# changez le nom à chaque fois svp ↓
experiment_name = "INATURALIST_E25_Mobilenet8142_D8142Relu_D8142Sigmoids_Lr0.3"
tb_callback = TensorBoard("./logs/" + experiment_name, )

print("Model training will start soon")

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# build the VGG16 network
model = applications.MobileNet(weights='imagenet', input_shape=input_shape, classes=8142, include_top=False, alpha=1.0,
                               depth_multiplier=1, dropout=1e-3)
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Flatten(input_shape=model.output_shape[1:])(model.output)
top_model = Dense(8142, activation='relu')(top_model)
top_model = Dropout(0.5)(top_model)
top_model = Dense(8142, activation='sigmoid')(top_model)
super_model = Model(model.input, top_model)

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning

# add the model on top of the convolutional base
# model.add(top_model)
# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
super_model.compile(loss=categorical_crossentropy,
                    optimizer=optimizers.SGD(lr=0.3, momentum=0.9),
                    metrics=[categorical_accuracy])

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

super_model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=[tb_callback],
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

super_model.save_weights('mobilenet.h5')
