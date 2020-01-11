import os
from keras.optimizers import SGD, RMSprop, Adam, Adadelta
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Reshape, Flatten, InputLayer, MaxPooling2D, UpSampling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.backend.tensorflow_backend import clear_session
from keras.callbacks import ModelCheckpoint
import pickle
import yaml

# define training images path
images_path = os.path.join(os.path.curdir, 'auto_images')
train_path = os.path.join(images_path, 'train')
summary_path = os.path.join(os.path.curdir, 'summary.txt')

#
train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.3, horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    os.path.abspath(train_path),
    target_size=(200, 200),
    batch_size=32, class_mode='input', subset='training')
valid_generator = train_datagen.flow_from_directory(os.path.abspath(train_path), target_size=(200, 200),
                                                    batch_size=32, class_mode='input', subset='validation')


def adam(lr):
    return Adam(learning_rate=lr)


def sgd(lr, momentum):
    return SGD(learning_rate=lr, momentum=momentum)


def rmsprop(lr):
    return RMSprop(learning_rate=lr)


def adadelta(lr):
    return Adadelta(learning_rate=lr)


optimizers = {
    'adam': adam,
    'sgd': sgd,
    'rmsprop': rmsprop,
    'adadelta': adadelta
}


def get_model():
    return Sequential([
        # encoder starts here
        InputLayer(input_shape=(200, 200, 3)),
        Conv2D(16, (4, 4), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Flatten(),
        Dense(1000, activation='relu'),
        # bottleneck
        Dense(5000, activation='relu'),
        Reshape((25, 25, 8)),
        Conv2D(8, (4, 4), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(8, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(3, (3, 3), activation='relu', padding='same')
    ])


def create_trial_dir(_id):
    os.mkdir('trial {}'.format(_id))
    return os.path.join(os.path.curdir, 'trial {}'.format(_id))


def load_trial_config(config):
    hyperparameters = dict(epochs=trial.get('epochs'), loss=trial.get('loss'))
    optimizer = trial.get('optimizer')
    return hyperparameters, optimizer

def log_summary(_id, history):
    loss = history.get('val_loss')
    epoch, minimum = (loss.index(min(loss)) + 1), min(loss)
    with open(summary_path, 'a+') as sum_handler:
        sum_handler.write('Model {}: min validation loss = {} @ epoch= {}\n'.format(_id, minimum, epoch))




def run_trial(_id, hyperparams, optimizer_options):
    trial_dir = create_trial_dir(_id)
    model_path = os.path.join(trial_dir, 'model {}.h5'.format(_id))
    checkpoint = ModelCheckpoint(model_path, save_best_only=True)
    optimizer_type = optimizer_options.get('type')
    lr = optimizer_options.get('learning_rate')
    if optimizer_type == 'sgd':
        momentum = optimizer_options.get('momentum')
        optimizer = optimizers.get(optimizer_type)(lr, momentum)
    else:
        optimizer = optimizers.get(optimizer_type)(lr)
    loss = hyperparams.get('loss')
    epochs = hyperparams.get('epochs')
    model = get_model()
    model.compile(optimizer=optimizer, loss=loss)
    print('Start Training for Model {}'.format(_id))
    model.fit_generator(train_generator, epochs=epochs, validation_data=valid_generator, callbacks=[checkpoint])
    history = model.history.history
    history_path = os.path.join(trial_dir, 'history {}'.format(_id))
    log_summary(_id, history)
    with open(history_path, 'wb') as handler:
        pickle.dump(history, handler)
    clear_session()
    print('End Training for Model {} Successfully'.format(_id))


with open('trials.yml') as yml_handler:
    trials = yaml.load(yml_handler, Loader=yaml.FullLoader)

for _id, trial in trials.items():
    hyperparameters, optim = load_trial_config(trial)
    run_trial(_id, hyperparameters, optim)
