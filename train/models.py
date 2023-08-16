import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import regularizers
from keras.models import load_model
import numpy as np
from scipy.ndimage import median_filter
from itertools import chain
import gc
import traceback
from datetime import datetime
from keras.models import Model

from train import metrics, plots, utils
import config
import os
import random

MODELS_DIR = config.MODELS_DIR
# SEEDS PARA REPRODUZIR RESULTADOS
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def build_model_lstm(input_shape, output_layers=1, return_seq=False):

    """Generates CNN model

    :param input_shape (tuple): Shape of input set
    :return model: CNN model

    baseado em: https://ieeexplore.ieee.org/abstract/document/8462677
    """

    # build network topology
    model = keras.Sequential()

    # 2 LSTM layers
    model.add(keras.layers.LSTM(256, input_shape=input_shape, return_sequences=True, stateful=False))#, dropout=0.2))#, recurrent_regularizer='l1'))
    if return_seq:
            model.add(keras.layers.LSTM(256, return_sequences=True, stateful=False))#, dropout=0.2))#, recurrent_regularizer='l1'))
    else:
            model.add(keras.layers.LSTM(256, stateful=False))#, dropout=0.2))#, recurrent_regularizer='l1'))

    # output layer
    model.add(keras.layers.Dense(output_layers, activation='linear'))

    return model


def build_model_speech(input_shape, seq_lenght, output_layers=1, return_seq=False):

    """Generates CNN model

    :param input_shape (tuple): Shape of input set
    :return model: CNN model

    baseado em: https://ieeexplore.ieee.org/abstract/document/8462677

    """

    model = keras.Sequential()
    # 1st conv layer
    model.add(keras.layers.Conv2D(64, (1, 8), activation='relu', padding='same', input_shape=input_shape,
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)))
    model.add(keras.layers.MaxPooling2D(pool_size=(1,10), padding='same'))
    model.add(keras.layers.Dropout(0.5))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(128, (1, 6), activation='relu', padding='same',
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)))
    model.add(keras.layers.MaxPooling2D(pool_size=(1,8), padding='same'))
    model.add(keras.layers.Dropout(0.5))

    # 3rd conv layer
    model.add(keras.layers.Conv2D(256, (1, 6), activation='relu', padding='same',
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
    bias_regularizer=regularizers.l2(1e-4),
    activity_regularizer=regularizers.l2(1e-5)))
    model.add(keras.layers.MaxPooling2D(pool_size=(1,8), padding='same'))
    model.add(keras.layers.Dropout(0.5))

    # reshape
    model.add(tf.keras.layers.Reshape((seq_lenght, 256)))

    # 2 LSTM layers
    model.add(keras.layers.LSTM(256, return_sequences=True, stateful=False))#, dropout=0.2))#, recurrent_regularizer='l1'))
    if return_seq:
          model.add(keras.layers.LSTM(256, return_sequences=True, stateful=False))#, dropout=0.2))#, recurrent_regularizer='l1'))
    else:
          model.add(keras.layers.LSTM(256, stateful=False))#, dropout=0.2))#, recurrent_regularizer='l1'))

    # output layer
    model.add(keras.layers.Dense(output_layers, activation='linear'))

    return model


def build_model_pretrained(input_shape, modelo, output_layers=1, return_seq=False):

    """Generates CNN model

    :param input_shape (tuple): Shape of input set
    :return model: CNN model

    baseado em: https://medium.com/smileinnovation/how-to-work-with-time-distributed-data-in-a-neural-network-b8b39aa4ce00
    https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f
    models: https://www.tensorflow.org/api_docs/python/tf/keras/applications
    https://www.tensorflow.org/tutorials/images/data_augmentation?hl=pt-br
    """

    """
    resize_and_rescale = tf.keras.Sequential([
      keras.layers.Rescaling(1./255)
    ])
    """

    """
    data_augmentation = tf.keras.Sequential([
      keras.layers.RandomFlip("horizontal"),
      keras.layers.RandomRotation(0.2),
      tf.keras.layers.RandomContrast(0.2),
      #tf.keras.layers.RandomTranslation(0.2),
      tf.keras.layers.RandomZoom(0.1)
    ])
    """

    model = tf.keras.Sequential()

    if modelo == "resnet":
          pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                            input_shape=(input_shape[1], input_shape[2], input_shape[3]),
                            pooling='avg',
                            weights='imagenet')
    elif modelo == "densenet":
          pretrained_model= tf.keras.applications.DenseNet201(include_top=False,
                            input_shape=(input_shape[1], input_shape[2], input_shape[3]),
                            pooling='avg',
                            weights='imagenet')
    elif modelo == "efficientnet":
          pretrained_model= tf.keras.applications.efficientnet_v2.EfficientNetV2B3(include_top=False,
                            input_shape=(input_shape[1], input_shape[2], input_shape[3]),
                            pooling='avg',
                            weights='imagenet')
    elif modelo == "convnext":
          pretrained_model= tf.keras.applications.convnext.ConvNeXtBase(include_top=False,
                            input_shape=(input_shape[1], input_shape[2], input_shape[3]),
                            pooling='avg',
                            weights='imagenet')
    elif modelo == "vgg19":
          pretrained_model= tf.keras.applications.vgg19.VGG19(include_top=False,
                            input_shape=(input_shape[1], input_shape[2], input_shape[3]),
                            pooling='avg',
                            weights='imagenet')

    """
    # Keep 10 layers to train
    trainable = 10
    for layer in pretrained_model.layers[:-trainable]:
        layer.trainable = False
    for layer in pretrained_model.layers[-trainable:]:
        layer.trainable = True
    """

    for layer in pretrained_model.layers:
        layer.trainable = False

    # add vgg model for 5 input images (keeping the right shape
    model.add(
      keras.layers.TimeDistributed(pretrained_model, input_shape=input_shape)
    )

    # now, flatten on each output to send 5
    # outputs with one dimension to LSTM
    model.add(
        keras.layers.TimeDistributed(
            keras.layers.Flatten()
        )
    )

    #model.add(keras.layers.AveragePooling1D())

    # 2 LSTM layers
    model.add(keras.layers.LSTM(512, return_sequences=True, stateful=False))#, dropout=0.2))#, recurrent_regularizer='l1'))
    if return_seq:
          model.add(keras.layers.LSTM(256, return_sequences=True, stateful=False))#, dropout=0.2))#, recurrent_regularizer='l1'))
    else:
          model.add(keras.layers.LSTM(256, stateful=False))#, dropout=0.2))#, recurrent_regularizer='l1'))

    # output layer
    model.add(keras.layers.Dense(output_layers, activation='linear'))

    return model


def create_model(input_shape, teste, seq_lenght, return_seq, modelo, print_summary=True):

    # create network
    if modelo == 'lstm':
        model = build_model_lstm(input_shape, return_seq=return_seq)
    elif modelo == 'speech':
        model = build_model_speech(input_shape, seq_lenght, return_seq=return_seq)
    else:
        model = build_model_pretrained(input_shape, modelo, return_seq=return_seq)

    # compile model
    #opt = keras.optimizers.Adam(learning_rate=0.0001)
    #model.compile(optimizer=opt, loss=concordance_loss)
    model.compile(optimizer="RMSprop", loss=metrics.concordance_loss)

    #teste model shapes
    teste = teste[np.newaxis, ...] # array shape (1, 150, 88, 1)
    x = model.predict(teste, verbose=0)

    if print_summary:
        utils.my_print("\n")
        utils.my_print(utils.get_model_summary(model))
        utils.my_print("\n")

    return model


def load_model(checkpoint_path, input_shape, teste, seq_lenght, modelo, return_seq):

    #load and save best_model
    # The model weights (that are considered the best) are loaded into the model.
    best_model = create_model(input_shape, teste, seq_lenght, return_seq, modelo, print_summary=False)
    best_model.load_weights(checkpoint_path)
    best_model.save(checkpoint_path / 'best-model-speech.h5')  # creates a HDF5 file 'my_model.h5'
    best_model.save_weights(checkpoint_path / 'best-model-weights-speech.h5')

    return best_model


def save_model(model, checkpoint_path, input_shape, teste, seq_lenght, modelo, return_seq):

    #save_last_model
    model.save(checkpoint_path / 'model.h5')  # creates a HDF5 file 'my_model.h5'
    model.save_weights(checkpoint_path / 'model-weights.h5')
    #del model  # deletes the existing model

    #load and save best_model
    # The model weights (that are considered the best) are loaded into the model.
    best_model = create_model(input_shape, teste, seq_lenght, return_seq, modelo, print_summary=False)
    best_model.load_weights(checkpoint_path)
    best_model.save(checkpoint_path / 'best-model-speech.h5')  # creates a HDF5 file 'my_model.h5'
    best_model.save_weights(checkpoint_path / 'best-model-weights-speech.h5')

    return best_model


def predict(best_model, X_test, y_test, output_layers, seq_lenght, plot=False, ds=False, validation=False):

    prediction = best_model.predict(X_test,verbose=0)
    flatten_prediction = list(chain.from_iterable(prediction))
    flatten_prediction = np.array(flatten_prediction)
    flatten_prediction = flatten_prediction.reshape(flatten_prediction.shape[0], output_layers)
    pred = flatten_prediction[:, 0]

    flatten_y = list(chain.from_iterable(y_test))
    flatten_y = np.array(flatten_y)
    flatten_y = flatten_y.reshape(flatten_y.shape[0], output_layers)
    actual = flatten_y[:, 0]

    median_pred = median_filter(pred, size=50, cval=0, mode='constant')

    if plot:
        for i in range (int(len(actual)/(7500-seq_lenght))):
            plots.plot_prediction(median_pred[(7500-seq_lenght)*i:(7500-seq_lenght)*(i+1)], actual[(7500-seq_lenght)*i:(7500-seq_lenght)*(i+1)], str(i+1))

    if not validation:
        #Calculate CCC
        result = metrics.concordance_cc(pred, actual)
        result2 = metrics.concordance_cc(median_pred, actual)
        utils.my_print("CCC (CONCORDANCE CORRELATION COEFFICIENT)")
        utils.my_print(f"\nCCC SEM Median Filter: {result}")
        utils.my_print(f"\nCCC COM Median Filter: {result2}")

    if ds:
        return utils.pred_lstm_data_transform(median_pred, seq_lenght)


def train_model(features, labels, arquitetura, seq_lenght, output_layers=1, texto='', return_seq=False):

    # train model
    t_inicial = datetime.now()
    datahora = t_inicial.strftime("%Y%m%d-%H%M%S")
    checkpoint_path = MODELS_DIR / f"models-{datahora}"

    utils.my_print("#"*120)
    utils.my_print(f"\n{texto}")
    utils.my_print(f"seq_lenght, return_seq: {seq_lenght}, {return_seq}\n")

    #train/validation/test split
    X_train, X_validation, X_test, y_train, y_validation, y_test = utils.prepare_datasets(features, labels, 3, 3, output_layers)
    utils.my_print("\ntrain/validation/test split")
    utils.my_print(f"X_train, X_validation, X_test: {X_train.shape}, {X_validation.shape}, {X_test.shape}")
    utils.my_print(f"y_train, y_validation, y_test: {y_train.shape}, {y_validation.shape}, {y_test.shape}")
    utils.my_print("\n")

    X_train, y_train = utils.lstm_data_transform(X_train, y_train, seq_lenght, return_sequences=return_seq)
    X_validation, y_validation = utils.lstm_data_transform(X_validation, y_validation, seq_lenght, return_sequences=return_seq)
    X_test, y_test = utils.lstm_data_transform(X_test, y_test, seq_lenght, to_shuffle=False, return_sequences=return_seq)

    utils.my_print("\nseq_lenght lstm_data_transform")
    utils.my_print(f"X_train, X_validation, X_test: {X_train.shape}, {X_validation.shape}, {X_test.shape}")
    utils.my_print(f"y_train, y_validation, y_test: {y_train.shape}, {y_validation.shape}, {y_test.shape}")
    utils.my_print("\n")

    input_shape = (X_train.shape[1], X_train.shape[2])
    if arquitetura == 'speech':
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
    if X_train.shape[3] != 1:
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    if len(X_train.shape) == 5 and X_train.shape[4] != 1:
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
    teste = X_train[1]

    model = create_model(input_shape, teste, seq_lenght, return_seq, arquitetura)

    #Replace NaN with zero and infinity with large finite numbers (default behaviour)
    #may be duplicated
    X_train = tf.convert_to_tensor(np.nan_to_num(X_train))
    X_validation = tf.convert_to_tensor(np.nan_to_num(X_validation))
    X_test = tf.convert_to_tensor(np.nan_to_num(X_test))
    y_train = tf.convert_to_tensor(y_train)
    y_validation = tf.convert_to_tensor(y_validation)
    y_test = tf.convert_to_tensor(y_test)

    backup_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, monitor = 'val_loss', verbose=0, save_best_only=True, mode='min')
    history = model.fit(X_train, y_train, validation_data=(X_validation, y_validation), batch_size=32, epochs=1, callbacks=[backup_callback], verbose=0)

    try:
        # plot accuracy/error for training and validation
        #plots.plot_history(history)
        del history
        del backup_callback
        del X_train, X_validation, y_train, y_validation
        gc.collect()

        best_model = save_model(model, checkpoint_path, input_shape, teste, seq_lenght, arquitetura, return_seq)
        del model
        gc.collect()

        if return_seq: seq_lenght = 0
        predict(best_model, X_test, y_test, output_layers, seq_lenght)
        utils.my_print(f"\nCHECKPOINT_PATH: {checkpoint_path}\n")
        utils.my_print("#"*120)
        utils.my_print("\n")

        del best_model
        del X_test, y_test

    except:
        print(traceback.format_exc())
        del X_test, y_test
        del model

    keras.backend.clear_session()
    gc.collect()


def teste_checkpoint(features, labels, seq_lenght, modelo, checkpoint_path, output_layers=1, return_seq=False):

    #train/validation/test split
    X_train, X_validation, X_test, y_train, y_validation, y_test = utils.prepare_datasets(features, labels, 3, 3, output_layers)
    X_train, y_train = utils.lstm_data_transform(X_train, y_train, seq_lenght, return_sequences=return_seq)
    X_test, y_test = utils.lstm_data_transform(X_test, y_test, seq_lenght, return_sequences=return_seq)

    #tf.convert_to_tensor()
    print("seq_lenght lstm_data_transform")
    print("X_train, X_test: ", X_train.shape, X_test.shape)
    print("y_train, y_test: ", y_train.shape, y_test.shape)
    print()

    input_shape = (X_train.shape[1], X_train.shape[2])
    if X_train.shape[3] != 1:
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    teste = X_train[1]

    del X_train, X_validation, y_train, y_validation
    gc.collect()

    best_model = load_model(checkpoint_path, input_shape, teste, seq_lenght, modelo, return_seq)

    X_test = tf.convert_to_tensor(np.nan_to_num(X_test))
    y_test = tf.convert_to_tensor(y_test)
    predict(best_model, X_test, y_test, output_layers, seq_lenght)
    print("CHECKPOINT_PATH: ", checkpoint_path)
    print("#"*120)

    del best_model
    del X_test, y_test
    keras.backend.clear_session()
    gc.collect()


def load_model_ds(features, labels, arquitetura, seq_lenght, checkpoint_path, output_cut, return_seq=False, output_layers=1, texto='', to_summary=False, to_predict=False, new_features=False, saved_information=False):

    print("#"*120)
    print(texto)
    print(checkpoint_path)

    #train/validation/test split
    X_train, X_validation, X_test, y_train, y_validation, y_test = utils.prepare_datasets(features, labels, 3, 3, output_layers)

    print("train/validation/test split")
    print("X_train, X_validation, X_test: ", X_train.shape, X_validation.shape, X_test.shape)
    print("y_train, y_validation, y_test: ", y_train.shape, y_validation.shape, y_test.shape)
    print()

    X_train, y_train = utils.lstm_data_transform(X_train, y_train, seq_lenght, to_shuffle=False, return_sequences=return_seq)
    X_validation, y_validation = utils.lstm_data_transform(X_validation, y_validation, seq_lenght, to_shuffle=False, return_sequences=return_seq)
    X_test, y_test = utils.lstm_data_transform(X_test, y_test, seq_lenght, to_shuffle=False, return_sequences=return_seq)

    print("seq_lenght lstm_data_transform")
    print("X_train, X_validation, X_test: ", X_train.shape, X_validation.shape, X_test.shape)
    print("y_train, y_validation, y_test: ", y_train.shape, y_validation.shape, y_test.shape)
    print()

    input_shape = (X_train.shape[1], X_train.shape[2])
    if arquitetura == 'speech':
        input_shape = (X_train.shape[1], X_train.shape[2], 1)
    if X_train.shape[3] != 1:
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    if len(X_train.shape) == 5 and X_train.shape[4] != 1:
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3], X_train.shape[4])
    teste = X_train[1]

    #Replace NaN with zero and infinity with large finite numbers (default behaviour)
    #may be duplicated
    X_train = np.nan_to_num(X_train)
    X_validation = np.nan_to_num(X_validation)
    X_test = np.nan_to_num(X_test)

    #load and save best_model
    # The model weights (that are considered the best) are loaded into the model.
    final_model = create_model(input_shape, teste, seq_lenght, return_seq, arquitetura, print_summary=False)

    if to_summary:
      final_model.summary()

    if new_features:
        final_model = Model(inputs=final_model.input, outputs=final_model.layers[output_cut].output)

    final_model.load_weights(checkpoint_path)

    if to_predict:
        prediction = predict(final_model, X_test, y_test, output_layers, seq_lenght, plot=False, ds=True)
        validation = predict(final_model, X_validation, y_validation, output_layers, seq_lenght, plot=False, validation=True, ds=True)
        print(prediction.shape, validation.shape)
        print("CHECKPOINT_PATH: ", checkpoint_path)
        print("#"*120)

        #X_train = np.array(list(chain.from_iterable(X_train)))
        #X_train = np.array(list(chain.from_iterable(X_train)))
        
        #X_validation = np.array(list(chain.from_iterable(X_validation)))
        #X_validation = np.array(list(chain.from_iterable(X_validation)))

        #X_test = np.array(list(chain.from_iterable(X_test)))
        #X_test = np.array(list(chain.from_iterable(X_test)))

        #print("list(chain.from_iterable")
        #print("X_train, X_validation, X_test: ", X_train.shape, X_validation.shape, X_test.shape)
        #print("y_train, y_validation, y_test: ", y_train.shape, y_validation.shape, y_test.shape)
        #print()

        if not saved_information:
            return X_train, X_validation, X_test, final_model, prediction, validation

        #depois da primeira execução já temos os dados salvos e só precisamos do modelo 
        return [], [], [], final_model, [], []

    if new_features:
        X_train = np.nan_to_num(X_train)
        X_validation = np.nan_to_num(X_validation)
        X_test = final_model.predict(X_test,verbose=0)[..., np.newaxis]
        X_validation = final_model.predict(X_validation,verbose=0)[..., np.newaxis]
        X_train = final_model.predict(X_train,verbose=0)[..., np.newaxis]
        print("X_train, X_validation, X_test: ", X_train.shape, X_validation.shape, X_test.shape)
        print("y_train, y_validation, y_test: ", y_train.shape, y_validation.shape, y_test.shape)

    if not saved_information:
        return X_train, X_validation, X_test, final_model, [], []
    #depois da primeira execução já temos os dados salvos e só precisamos do modelo 
    return [], [], [], final_model, [], []
