from keras.callbacks import *
from keras.layers import *
from keras.metrics import *
from keras.models import *
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
learning_rate = 0.1
dense_t = 64
nb_classe_sortie = 8142
activ = 'relu'
final_activ = 'softmax'
losses = 'categorical_crossentropy'
optimiz = 'sgd'
i = 1

# changez le nom à chaque fois svp ↓
experiment_name = "INATURALIST[" + str(i) + "]_E[" + str(epochs) + "]_S[" + str(
    nb_classe_sortie) + "]_FA[" + final_activ + "]_BS[" + str(batch_size) + "]_LR[" + str(learning_rate) + "]_D[" + str(
    dense_t) + "]_A[" + activ + "]_L[" + losses + "]_O[" + optimiz + "]"
tb_callback = TensorBoard("./logs/" + experiment_name, )

print("Model training will start soon")

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

while i < 20:
    # change batch size
    if i == 1:
        batch_size = 128
    if i == 2:
        batch_size = 256
    if i == 3:
        batch_size = 512
    if i == 4:
        batch_size = 1024
    # change learning rate
    if i == 5:
        learning_rate = 0.2
    if i == 6:
        learning_rate = 0.5
    # change dense
    if i == 7:
        dense_t = 128
    if i == 8:
        dense_t = 256
    # change activation
    # enlever active si modéle lineaire
    if i == 9:
        activ = 'hard_sigmoid'
    if i == 10:
        activ = 'selu'
    if i == 11:
        activ = 'elu'
    if i == 12:
        activ = 'linear'
    if i == 13:
        activ = 'tanh'
    # change loss
    if i == 14:
        activ = 'relu'
        losses = 'mse'
    if i == 15:
        losses = 'binary_crossentropy'
    # change optimizer
    if i == 16:
        losses = 'categorical_crossentropy'
        optimiz = 'rmsprop'
    if i == 17:
        optimiz = 'adagrad'
    if i == 18:
        optimiz = 'adadelta'
    if i == 19:
        optimiz = 'tfoptimizer'
    i += 1
    experiment_name = "INATURALIST[" + str(i) + "]_E[" + str(epochs) + "]_S[" + str(
        nb_classe_sortie) + "]_FA[" + final_activ + "]_BS[" + str(batch_size) + "]_LR[" + str(
        learning_rate) + "]_D[" + str(dense_t) + "]_A[" + activ + "]_L[" + losses + "]_O[" + optimiz + "]"

    model = Sequential()
    model.add(Dense(8142, input_shape=input_shape))
    model.add(Activation('sigmoid'))

    model.compile(sgd(lr=learning_rate), losses, metrics=[categorical_accuracy])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,

        target_size=(img_width, img_height),
        batch_size=batch_size, shuffle=True, seed=None,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size, shuffle=True, seed=None,
        class_mode='categorical')

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        callbacks=[tb_callback],
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    model.save_weights('first_try.h5')
