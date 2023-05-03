import config
import numpy as np
import pickle

from sklearn.utils import shuffle

RECOLA_PICKLE_PATH = config.RECOLA_PICKLE_PATH
SEGMENTS_PER_PERSON = config.SEGMENTS_PER_PERSON
RECOLA_SUBJECTS = config.RECOLA_SUBJECTS

import os
import random

import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = config.MODELS_DIR
# SEEDS PARA REPRODUZIR RESULTADOS
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def load_data(path):
    with open(path, "rb") as fp:
        data = pickle.load(fp)
    return data

def prepare_datasets(X, y, validation_size, test_size, output_layers):

    """Loads data and splits it into train, validation and test sets.
    :param test_size (int): Número de pessoas of data set to allocate to test split
    :param validation_size (int): Número de pessoas of data set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set

    https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
    """

    train_split = SEGMENTS_PER_PERSON * (RECOLA_SUBJECTS - validation_size - test_size) 
    validation_split = SEGMENTS_PER_PERSON * validation_size
    test_split = SEGMENTS_PER_PERSON * test_size

    #não é possível usar train_test_split com porcentagens para garantir sequencias da mesma pessoa
    #create train, validation and test split
    X_test, y_test =  X[:test_split], y[:test_split]
    X_validation, y_validation = X[test_split:test_split+validation_split], y[test_split:test_split+validation_split]
    X_train, y_train = X[test_split+validation_split:train_split+validation_split+test_split], y[test_split+validation_split:train_split+validation_split+test_split]

    """
    # fit scaler on training data
    norm = MinMaxScaler().fit(np.nan_to_num(X_train))
    # transform training data
    X_train = norm.transform(X_train)
    # transform testing dataabs
    X_test = norm.transform(X_test)
    # transform testing dataabs
    X_validation = norm.transform(X_validation)
    """
    
    #reshape data
    if len(X_train.shape) == 3:
      X_train = np.reshape(X_train, (-1, X_train.shape[1], X_train.shape[2]))
    elif len(X_train.shape) == 4:
      X_train = np.reshape(X_train, (-1, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    else:
      X_train = np.reshape(X_train, (-1, X_train.shape[1], 1))
    y_train = np.reshape(y_train, (-1, output_layers))

    if len(X_validation.shape) == 3:
      X_validation = np.reshape(X_validation, (-1, X_validation.shape[1], X_validation.shape[2]))
    elif len(X_validation.shape) == 4:
      X_validation = np.reshape(X_validation, (-1, X_validation.shape[1], X_validation.shape[2], X_validation.shape[3]))
    else:
      X_validation = np.reshape(X_validation, (-1, X_validation.shape[1], 1))
    y_validation = np.reshape(y_validation, (-1, output_layers))

    if len(X_test.shape) == 3:
      X_test = np.reshape(X_test, (-1, X_test.shape[1], X_test.shape[2]))
    elif len(X_test.shape) == 4:
      X_test = np.reshape(X_test, (-1, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
    else:
      X_test = np.reshape(X_test, (-1, X_test.shape[1], 1))
    y_test = np.reshape(y_test, (-1, output_layers))

    #shuffle será realizado após lstm_data_transform de acordo com sequence lenght
    #X_train = shuffle(X_train, random_state=0)
    #y_train = shuffle(y_train, random_state=0)
    #X_validation = shuffle(X_validation, random_state=0)
    #y_validation = shuffle(y_validation, random_state=0)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def prepare_datasets2(X, y, validation_size, test_size, output_layers):

    """Loads data and splits it into train, validation and test sets.
    :param test_size (int): Número de pessoas of data set to allocate to test split
    :param validation_size (int): Número de pessoas of data set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set

    https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/
    """

    train_split = SEGMENTS_PER_PERSON * (RECOLA_SUBJECTS - validation_size - test_size) 
    validation_split = SEGMENTS_PER_PERSON * validation_size
    test_split = SEGMENTS_PER_PERSON * test_size

    #não é possível usar train_test_split com porcentagens para garantir sequencias da mesma pessoa
    #create train, validation and test split
    X_test, y_test =  X[:test_split], y[:test_split]
    X_validation, y_validation = X[test_split:test_split+validation_split], y[test_split:test_split+validation_split]
    X_train, y_train = X[test_split+validation_split:train_split+validation_split+test_split], y[test_split+validation_split:train_split+validation_split+test_split]

    """
    # fit scaler on training data
    norm = MinMaxScaler().fit(np.nan_to_num(X_train))
    # transform training data
    X_train = norm.transform(X_train)
    # transform testing dataabs
    X_test = norm.transform(X_test)
    # transform testing dataabs
    X_validation = norm.transform(X_validation)
    """
    
    #reshape data
    if len(X_train.shape) == 3:
        X_train = np.reshape(X_train, (-1, X_train.shape[1], X_train.shape[2]))
    elif len(X_train.shape) == 4:
        X_train = np.reshape(X_train, (-1, X_train.shape[1], X_train.shape[2], X_train.shape[3]))
    else:
       X_train = np.reshape(X_train, (-1, X_train.shape[1], 1))
    y_train = np.reshape(y_train, (-1, output_layers))

    if len(X_validation.shape) == 3:
         X_validation = np.reshape(X_validation, (-1, X_validation.shape[1], X_validation.shape[2]))
    elif len(X_validation.shape) == 4:
         X_validation = np.reshape(X_validation, (-1, X_validation.shape[1], X_validation.shape[2], X_validation.shape[3]))
    else:
         X_validation = np.reshape(X_validation, (-1, X_validation.shape[1], 1))
    y_validation = np.reshape(y_validation, (-1, output_layers))

    if len(X_test.shape) == 3:
          X_test = np.reshape(X_test, (-1, X_test.shape[1], X_test.shape[2]))
    elif len(X_test.shape) == 4:
         X_test = np.reshape(X_test, (-1, X_test.shape[1], X_test.shape[2], X_test.shape[3]))
    else:
        X_test = np.reshape(X_test, (-1, X_test.shape[1], 1))
    y_test = np.reshape(y_test, (-1, output_layers))

    #shuffle será realizado após lstm_data_transform de acordo com sequence lenght
    #X_train = shuffle(X_train, random_state=0)
    #y_train = shuffle(y_train, random_state=0)
    #X_validation = shuffle(X_validation, random_state=0)
    #y_validation = shuffle(y_validation, random_state=0)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


def lstm_data_transform(x_data, y_data, seq_lenght, return_sequences=False, to_shuffle=False):

    """ Changes data to the format for LSTM training for sequence lenght split approach """

    # Prepare the list for the transformed data
    X, y = list(), list()

    # Loop of the entire data set
    # 7500 * 0.04 = 300s = 5min

    for i in range(int(x_data.shape[0]/7500)):

        item = x_data[7500*i : 7500*(i+1)]  
        label = y_data[7500*i : 7500*(i+1)]  
        #print(i, item.shape, label.shape)

        j = 0

        while j < item.shape[0]:

            # compute a new (sliding window) index
            end_ix = j + seq_lenght
            # if index is larger than the size of the dataset, we stop
            if ((end_ix > item.shape[0]) or (not return_sequences and end_ix == item.shape[0])):
                break

            # Get a sequence of data for x
            seq_X = item[j:end_ix]

            if return_sequences:
                # Get label for all elements
                seq_y = label[j:end_ix]
                j = end_ix
            else:
                # Get only the last element of the sequency for y
                seq_y = label[end_ix]
                j += 1

            # Append the list with sequencies
            X.append(seq_X)
            y.append(seq_y)

    X = np.array(X)
    y = np.array(y)

    if to_shuffle:
        print('blblblblbllblblblblblblblblblblbllblblblb')
        X = shuffle(X, random_state=0)
        y = shuffle(y, random_state=0)

    return X, y


def pred_lstm_data_transform(y_data, seq_lenght, return_sequences=True):

    """ Changes data to the format for LSTM training for sequence lenght split approach """

    # Prepare the list for the transformed data
    y = list()

    # Loop of the entire data set
    # 7500 * 0.04 = 300s = 5min

    for i in range(int(y_data.shape[0]/7500)):

      label = y_data[7500*i : 7500*(i+1)]  

      j = 0

      while j < label.shape[0]:

        # compute a new (sliding window) index
        end_ix = j + seq_lenght
        # if index is larger than the size of the dataset, we stop
        if ((end_ix > label.shape[0]) or (not return_sequences and end_ix == label.shape[0])):
            break

        if return_sequences:
            # Get label for all elements
            seq_y = label[j:end_ix]
            j = end_ix
        else:
            # Get only the last element of the sequency for y
            seq_y = label[end_ix]
            j += 1

        # Append the list with sequencies
        y.append(seq_y)

    y = np.array(y)

    return y
