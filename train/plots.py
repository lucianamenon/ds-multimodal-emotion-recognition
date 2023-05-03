import matplotlib.pyplot as plt
import config

MODELOS = config.MODELOS

def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    # create error sublpot
    plt.plot(history.history["loss"], label="train error")
    plt.plot(history.history["val_loss"], label="validation error")
    plt.ylabel("CCC")
    plt.xlabel("Epoch")
    plt.legend(loc="upper right")
    plt.title("CCC eval")

    plt.show()


def plot_prediction(pred, actual, example):

    # plot prediction per person
    plt.figure(figsize=(15,5))
    plt.plot(pred, label="prediction")
    plt.plot(actual, label="gold standard")
    plt.xlabel("Time")
    plt.legend(loc="upper right")
    plt.title("Prediction caso de teste - " + example)

    plt.show()


def plot_all(data, label):

    plt.figure(figsize=(20,6))
    #flatten_data = list(chain.from_iterable(data))
    #flatten_Y_test = list(chain.from_iterable(Y_test))
    for i in range(18):
        item = data[7500*i : 7500*(i+1)]
        plt.plot(item, label=f'audio_{i+1}')
    #plt.xlabel('Time (minutes)', fontsize=16)
    plt.ylabel(label)
    #plt.xlabel("Time")

    plt.legend(loc="upper right")

    plt.show()


def plot_all_models(data, label, person):

    # plot prediction per model
    plt.figure(figsize=(20,6))
    #flatten_data = list(chain.from_iterable(data))
    #flatten_Y_test = list(chain.from_iterable(Y_test))
    for i in range(len(data)):
        modelo = data[i]
        item = modelo[7500*person : 7500*(person+1)]
        plt.plot(item, label=MODELOS[i])
    #plt.xlabel('Time (minutes)', fontsize=16)
    plt.ylabel(label)
    #plt.xlabel("Time")

    plt.legend(loc="upper right")

    plt.show()


def plot_one(X, label='signal', ylabel=''):

    # plot prediction per person
    plt.figure(figsize=(15,5))
    plt.plot(X, label=label)
    plt.legend(loc="upper right")
    plt.ylabel(ylabel)

    plt.show()
