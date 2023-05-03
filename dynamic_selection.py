import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
from itertools import chain
from scipy.spatial import distance
import math

from train import utils, plots, models, metrics
import config
import random
import os
MODELOS = config.MODELOS

import warnings
warnings.filterwarnings('ignore')

# SEEDS PARA REPRODUZIR RESULTADOS - QUALQUER NUMERO
RANDOM_SEED = 42

# PYTHONHASHSEED environment variable
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
# python built-in pseudo-random generator
random.seed(RANDOM_SEED)
# numpy
np.random.seed(RANDOM_SEED)

def ds(k, seq_len, return_sequences, saved_information=False):

    ds_models, bases, preds, validation_preds, final_dist = load_data(seq_len, return_sequences, saved_information)
    flatten_y = list(chain.from_iterable(bases['y_test']))
    flatten_y = np.array(flatten_y)
    #flatten_y = flatten_y.reshape(22500, 1)
    y = flatten_y[:,0]

    media(y, preds)
    dinamic_selection(y, bases, ds_models, final_dist, preds, validation_preds, k, debug=True)


def peso_regressor(t, model, dist_t, args, preds, valitation_preds, y, output_shape=(50), threshold=0.5, debug=False):

    #dist_t: distancias do caso de teste t
    #args_raw: argumentos dos itens com as k menores distâncias
    #test_t: caso de teste t
    #preds: predição base teste
    #valitation_preds: predição base validação

    #predição caso de teste t
    pred_t = preds[t].reshape(output_shape)

    #somatório dos k itens com menor distância com o caso de teste t
    somatorio_item = 0
    sum_item = 0
    sum_error = 0

    for i, item in enumerate(args):
        somatorio_item += 1 / (dist_t[item]+1)

    somatorio_regressor = 0
    somatorio_regressor3 = 0
    for i, item in enumerate(args):
        #peso item
        peso_item = (1 / (dist_t[item]+1)) / somatorio_item
        #if debug: print(item, peso_item)

        #gold standard
        actual = y[item].reshape(output_shape)

        #predição caso de teste similar
        pred_v = valitation_preds[item]

        #if debug: print(actual.shape, pred_t.shape, pred_v.shape)

        #validation ccc
        #ccc = (concordance_cc(pred_v, actual)+1)/2 #aqui estamos considerando o valor de ccc normalizado entre 0 e 1
        ccc = metrics.concordance_cc(pred_v, actual)
        ccc_norm = (ccc+1)/2
        #print(ccc, 1-ccc, ccc_norm, 1-ccc_norm)
        if debug: print(item, dist_t[item], peso_item, ccc_norm, 1-ccc_norm)
        sum_item += peso_item
        sum_error += 1-ccc_norm

        somatorio_regressor += peso_item * (1-ccc_norm)

    #peso regresor - erro > threshold
    if 1-ccc_norm > threshold:
        somatorio_regressor3 += 1
    else:
        somatorio_regressor3 += peso_item * (1-ccc_norm)

    peso_regressor = 1/(math.sqrt(somatorio_regressor))
    peso_regressor_3 = 1/(math.sqrt(somatorio_regressor3))

    #if debug: print("sum error: ", sum_error, sum_error/len(args))
    if sum_error/len(args) > threshold: #no método de Dynamic Weighting with Selection (DWS) os regressores com o erro médio superior a um valor definido são descartados
        peso_regressor_2 = 0
    else:
        peso_regressor_2 = peso_regressor

    if debug:
        print(peso_regressor, peso_regressor_2, peso_regressor_3, sum_item)
        print()

    return somatorio_regressor, peso_regressor, peso_regressor_2, peso_regressor_3, pred_t


def dinamic_selection(y, bases, models, final_dist, test_preds, valitation_preds, k, debug=False):

    final_pred_selection = []
    final_pred_weight = []
    final_pred_weight_selection = []
    final_pred_weight_selection_2 = []
    dist_audio = final_dist['dist_audio']
    dist_ecg = final_dist['dist_ecg']
    dist_hrv = final_dist['dist_hrv']
    dist_video = final_dist['dist_video']
    dist_geometric = final_dist['dist_geometric']

    #if debug:
    #  print("np.asarray(dist_audio).shape", "np.asarray(dist_ecg).shape", "np.asarray(dist_hrv).shape", "np.asarray(dist_video).shape", "np.asarray(dist_geometric).shape")
    #  print(np.asarray(dist_audio).shape, np.asarray(dist_ecg).shape, np.asarray(dist_hrv).shape, np.asarray(dist_video).shape, np.asarray(dist_geometric).shape)
    #  print("bases['y_test'].shape")
    #  print(bases['y_test'].shape)

    for t, test in enumerate(bases['y_test']):

        #k vizinhos mais próximos de cada caso de teste - em cada view
        args_audio = np.argsort(dist_audio[t])[:k]
        args_ecg = np.argsort(dist_ecg[t])[:k]
        args_hrv = np.argsort(dist_hrv[t])[:k]
        args_video = np.argsort(dist_video[t])[:k]
        args_geometric = np.argsort(dist_geometric[t])[:k]

        if debug:
            print()
            print('#'*150)
            print("Caso de teste {}: \n".format(t+1))
        #  print("t", "args_audio.shape", "np.asarray(dist_audio[t]).shape", "np.asarray(test_preds['modelo_audio_arousal']).shape", "np.asarray(valitation_preds['modelo_audio_arousal']).shape", "np.asarray(bases['y_validation']).shape")
        #  print(t, args_audio.shape, np.asarray(dist_audio[t]).shape, np.asarray(test_preds['modelo_audio_arousal']).shape, np.asarray(valitation_preds['modelo_audio_arousal']).shape, np.asarray(bases['y_validation']).shape)
        
        #AUDIO FEATURES
        if debug: print('AUDIO FEATURES')
        error_audio, somatorio_audio, somatorio_audio_ws, somatorio_audio_ws_2, pred_audio = peso_regressor(t, models['modelo_audio_arousal'], dist_audio[t], args_audio, test_preds['modelo_audio_arousal'], valitation_preds['modelo_audio_arousal'], bases['y_validation'], debug=debug)

        #ECG FEATURES
        if debug: print('ECG FEATURES')
        error_ecg, somatorio_ecg, somatorio_ecg_ws, somatorio_ecg_ws_2, pred_ecg = peso_regressor(t, models['modelo_ecg_arousal'], dist_ecg[t], args_ecg, test_preds['modelo_ecg_arousal'], valitation_preds['modelo_ecg_arousal'], bases['y_validation'], debug=debug)

        #HRV FEATURES
        if debug: print('HRV FEATURES')
        error_hrv, somatorio_hrv, somatorio_hrv_ws, somatorio_hrv_ws_2, pred_hrv = peso_regressor(t, models['modelo_hrv_arousal'], dist_hrv[t], args_hrv, test_preds['modelo_hrv_arousal'], valitation_preds['modelo_hrv_arousal'], bases['y_validation'], debug=debug)

        #VIDEO FEATURES
        if debug: print('VIDEO FEATURES')
        error_video, somatorio_video, somatorio_video_ws, somatorio_video_ws_2, pred_video = peso_regressor(t, models['modelo_video_arousal'], dist_video[t], args_video, test_preds['modelo_video_arousal'], valitation_preds['modelo_video_arousal'], bases['y_validation'], debug=debug)

        #VIDEO GEOMETRIC FEATURES
        if debug: print('VIDEO GEOMETRIC FEATURE')
        error_geometric, somatorio_geometric, somatorio_geometric_ws, somatorio_geometric_ws_2, pred_geometric = peso_regressor(t, models['modelo_geometric_arousal'], dist_geometric[t], args_geometric, test_preds['modelo_geometric_arousal'], valitation_preds['modelo_geometric_arousal'], bases['y_validation'], debug=debug)

        if debug:
            print(somatorio_audio, somatorio_ecg, somatorio_hrv, somatorio_video, somatorio_geometric)
            print(somatorio_audio_ws, somatorio_ecg_ws, somatorio_hrv_ws, somatorio_video_ws, somatorio_geometric_ws)
            print(somatorio_audio_ws_2, somatorio_ecg_ws_2, somatorio_hrv_ws_2, somatorio_video_ws_2, somatorio_geometric_ws_2)

        preds = [pred_audio, pred_ecg, pred_hrv, pred_video, pred_geometric]
        errors = [error_audio, error_ecg, error_hrv, error_video, error_geometric]
        arg_min = np.argsort(errors)[0]

        somatorio_raizes = somatorio_audio + somatorio_ecg + somatorio_hrv + somatorio_video + somatorio_geometric
        peso_audio = somatorio_audio/somatorio_raizes
        peso_ecg = somatorio_ecg/somatorio_raizes
        peso_hrv = somatorio_hrv/somatorio_raizes
        peso_video = somatorio_video/somatorio_raizes
        peso_geometric = somatorio_geometric/somatorio_raizes

        somatorio_raizes_ws_2 = somatorio_audio_ws_2 + somatorio_ecg_ws_2 + somatorio_hrv_ws_2 + somatorio_video_ws_2 + somatorio_geometric_ws_2
        peso_audio_ws_2 = somatorio_audio_ws_2/somatorio_raizes_ws_2
        peso_ecg_ws_2 = somatorio_ecg_ws_2/somatorio_raizes_ws_2
        peso_hrv_ws_2 = somatorio_hrv_ws_2/somatorio_raizes_ws_2
        peso_video_ws_2 = somatorio_video_ws_2/somatorio_raizes_ws_2
        peso_geometric_ws_2 = somatorio_geometric_ws_2/somatorio_raizes_ws_2

        somatorio_raizes_ws = somatorio_audio_ws + somatorio_ecg_ws + somatorio_hrv_ws + somatorio_video_ws + somatorio_geometric_ws
        if somatorio_raizes_ws > 0: #tratativa caso todos os regressores tiverem abaixo do threshold
            peso_audio_ws = somatorio_audio_ws/somatorio_raizes_ws
            peso_ecg_ws = somatorio_ecg_ws/somatorio_raizes_ws
            peso_hrv_ws = somatorio_hrv_ws/somatorio_raizes_ws
            peso_video_ws = somatorio_video_ws/somatorio_raizes_ws
            peso_geometric_ws = somatorio_geometric_ws/somatorio_raizes_ws
        else:
            peso_audio_ws, peso_ecg_ws, peso_hrv_ws, peso_video_ws, peso_geometric_ws = 0, 0, 0, 0, 0

        if debug:
            print(peso_audio, peso_ecg, peso_hrv, peso_video, peso_geometric, peso_audio+peso_ecg+peso_hrv+peso_video+peso_geometric)
            print(peso_audio_ws, peso_ecg_ws, peso_hrv_ws, peso_video_ws, peso_geometric_ws, peso_audio_ws+peso_ecg_ws+peso_hrv_ws+peso_video_ws+peso_geometric_ws)
            print(peso_audio_ws_2, peso_ecg_ws_2, peso_hrv_ws_2, peso_video_ws_2, peso_geometric_ws_2, peso_audio_ws_2+peso_ecg_ws_2+peso_hrv_ws_2+peso_video_ws_2+peso_geometric_ws_2)
            print(errors[0], errors[1], errors[2], errors[3], errors[4], errors[arg_min], arg_min)
            print('#'*150)
            print()

        final_pred_selection.append(preds[arg_min])
        final_pred_weight.append(pred_audio*peso_audio + pred_ecg*peso_ecg + pred_hrv*peso_hrv + pred_video*peso_video + pred_geometric*peso_geometric)
        if somatorio_raizes_ws > 0:
            final_pred_weight_selection.append(pred_audio*peso_audio_ws + pred_ecg*peso_ecg_ws + pred_hrv*peso_hrv_ws + pred_video*peso_video_ws + pred_geometric*peso_geometric_ws)
        else: #todos os itens com peso zerado, pegamos o melhor resultado do método base DS
            final_pred_weight_selection.append(preds[arg_min])
        if somatorio_raizes_ws_2 > 0:
            final_pred_weight_selection_2.append(pred_audio*peso_audio_ws_2 + pred_ecg*peso_ecg_ws_2 + pred_hrv*peso_hrv_ws_2 + pred_video*peso_video_ws_2 + pred_geometric*peso_geometric_ws_2)
        else: #todos os itens com peso zerado, pegamos o melhor resultado do método base DS
            final_pred_weight_selection_2.append(preds[arg_min])

    print("\n\nDynamic Selection (DS)")
    final_prediction(y, final_pred_selection)

    print("\n\nDynamic Weighting (DW)")
    final_prediction(y, final_pred_weight)

    print("\n\nDynamic Weighting with Selection (DWS)")
    final_prediction(y, final_pred_weight_selection)

    print("\n\nDynamic Weighting with Selection (DWS) 2")
    final_prediction(y, final_pred_weight_selection_2)

    return


def final_prediction(y, final_pred, plot=True):

    ##Flatten prediction
    print(y.shape, len(final_pred))
    flatten_final_prediction = list(chain.from_iterable(final_pred))
    print(y.shape, len(flatten_final_prediction))
    flatten_final_prediction = np.array(flatten_final_prediction)
    print(y.shape, flatten_final_prediction.shape)

    ##Plot RESULTS
    if flatten_final_prediction.shape[0] == 22500:
        if plot:
            plots.plot_prediction(flatten_final_prediction[:7500], y[:7500], '1')
            result_t1 = metrics.concordance_cc(flatten_final_prediction[:7500], y[:7500])
            print(" - AROUSAL CCC T1: ", result_t1)
            plots.plot_prediction(flatten_final_prediction[7500:7500*2], y[7500:7500*2], '2')
            result_t2 = metrics.concordance_cc(flatten_final_prediction[7500:7500*2], y[7500:7500*2])
            print(" - AROUSAL CCC T2: ", result_t2)
            plots.plot_prediction(flatten_final_prediction[7500*2:7500*3], y[7500*2:7500*3], '3')
            result_t3 = metrics.concordance_cc(flatten_final_prediction[7500*2:7500*3], y[7500*2:7500*3])
            print(" - AROUSAL CCC T3: ", result_t3)

        ##Calculate AROUSAL CCC APÓS SELEÇÃO DINÂMICA
        result = metrics.concordance_cc(flatten_final_prediction, y)
        print("AROUSAL CCC (Concordance correlation coefficient): ", result)

    else:
        plots.plot_prediction(flatten_final_prediction[:7500], y[:7500], '1')
        result_t1 = metrics.concordance_cc(flatten_final_prediction[:7500], y[:7500])
        print(" - AROUSAL CCC T1: ", result_t1)


def media(y, preds):

    #plots.plot_one(y[:7500], label='Gold Standard - Test T1', ylabel='AROUSAL')
    #plots.plot_one(y[7500:7500*2], label='Gold Standard - Test T2', ylabel='AROUSAL')
    #plots.plot_one(y[7500*2:7500*3], label='Gold Standard - Test T3', ylabel='AROUSAL')

    #prediction = best_model.predict(X_test,verbose=0)
    flatten_audio = list(chain.from_iterable(preds['modelo_audio_arousal']))
    flatten_audio = np.array(flatten_audio)
    #flatten_audio = flatten_audio.reshape(22500)

    #prediction = best_model.predict(X_test,verbose=0)
    flatten_ecg = list(chain.from_iterable(preds['modelo_ecg_arousal']))
    flatten_ecg = np.array(flatten_ecg)
    #flatten_ecg = flatten_ecg.reshape(22500)

    #prediction = best_model.predict(X_test,verbose=0)
    flatten_hrv = list(chain.from_iterable(preds['modelo_hrv_arousal']))
    flatten_hrv = np.array(flatten_hrv)
    #flatten_hrv = flatten_hrv.reshape(22500)

    #prediction = best_model.predict(X_test,verbose=0)
    flatten_video = list(chain.from_iterable(preds['modelo_video_arousal']))
    flatten_video = np.array(flatten_video)
    #flatten_video = flatten_video.reshape(22500)

    #prediction = best_model.predict(X_test,verbose=0)
    flatten_geometric = list(chain.from_iterable(preds['modelo_geometric_arousal']))
    flatten_geometric = np.array(flatten_geometric)
    #flatten_geometric = flatten_geometric.reshape(22500)

    plots.plot_all_models([y, flatten_audio, flatten_ecg, flatten_hrv, flatten_video, flatten_geometric], 'AROUSAL TEST T1', 0)
    plots.plot_all_models([y, flatten_audio, flatten_ecg, flatten_hrv, flatten_video, flatten_geometric], 'AROUSAL TEST T2', 1)
    plots.plot_all_models([y, flatten_audio, flatten_ecg, flatten_hrv, flatten_video, flatten_geometric], 'AROUSAL TEST T3', 2)

    media = (flatten_audio + flatten_ecg + flatten_hrv + flatten_video + flatten_geometric) / 5

    for i in range (int(len(y)/7500)):
        plots.plot_prediction(media[7500*i:7500*(i+1)], y[7500*i:7500*(i+1)], str(i+1))

    #Calculate CCC
    result = metrics.concordance_cc(media, y)
    print("CCC (CONCORDANCE CORRELATION COEFFICIENT) - MÉDIA TODOS OS MODELOS")
    print("CCC COM Median Filter: ", result)

def load_data(seq_len, return_sequences, saved_information):

    data = utils.load_data(config.RECOLA_PICKLE_PATH)

    if seq_len == 50 and return_sequences == True:
        models_path = config.PATH_MODELOS_50_TRUE
        pickle_path = config.RECOLA_PICKLE_PATH_SPLIT_50_TRUE
        test_pred_path = config.RECOLA_TEST_PRED_PATH_50_TRUE
        validation_pred_path = config.RECOLA_VALIDATION_PRED_PATH_50_TRUE
        dist_path = config.RECOLA_VALIDATION_DIST_PATH_50_TRUE

    elif seq_len == 50 and return_sequences == False:
        models_path = config.PATH_MODELOS_50_FALSE
        pickle_path = config.RECOLA_PICKLE_PATH_SPLIT_50_FALSE
        test_pred_path = config.RECOLA_TEST_PRED_PATH_50_FALSE
        validation_pred_path = config.RECOLA_VALIDATION_PRED_PATH_50_FALSE
        dist_path = config.RECOLA_VALIDATION_DIST_PATH_50_FALSE

    #Acoustic features (eGeMAPS) - 88 features
    checkpoint_path = models_path[0]
    _, validation_audio_arousal, test_audio_arousal, modelo_audio_arousal, pred_audio_arousal, pred_validation_audio_arousal = models.load_model_ds(np.array(data["arousal_audio_feature"]), data["arousal_label"], "lstm", seq_len, checkpoint_path, -1, texto="Acoustic features (eGeMAPS) - 88 features\n", return_seq=return_sequences, to_summary=True, to_predict=True, new_features=False)

    #Eletrocardiograma (ECG) - 19 features
    checkpoint_path = models_path[1]
    _, validation_ecg_arousal, test_ecg_arousal, modelo_ecg_arousal, pred_ecg_arousal, pred_validation_ecg_arousal = models.load_model_ds(np.array(data["arousal_ecg"]), data["arousal_label"], "lstm", seq_len, checkpoint_path, -1, texto="Eletrocardiograma (ECG) - 19 features\n", return_seq=return_sequences, to_summary=True, to_predict=True, new_features=False)

    #Heart Rate First Order Derivate (HRV) - 10 features
    checkpoint_path = models_path[2]
    _, validation_hrv_arousal, test_hrv_arousal, modelo_hrv_arousal, pred_hrv_arousal, pred_validation_hrv_arousal = models.load_model_ds(np.array(data["arousal_hrv"]), data["arousal_label"], "lstm", seq_len, checkpoint_path, -1, texto="Heart Rate First Order Derivate (HRV) - 10 features\n", return_seq=return_sequences, to_summary=True, to_predict=True, new_features=False)

    #Appearance features (PCA from 50k LGBP-TOP) - 168 features
    checkpoint_path = models_path[3]
    _, validation_video_arousal, test_video_arousal, modelo_video_arousal, pred_video_arousal, pred_validation_video_arousal = models.load_model_ds(np.array(data["arousal_video_feature"]), data["arousal_label"], "lstm", seq_len, checkpoint_path, -1, texto="Appearance features (PCA from 50k LGBP-TOP) - 168 features\n", return_seq=return_sequences, to_summary=True, to_predict=True, new_features=False)

    #Geometric features (derived from 49 facial landmarks) - 632 features
    checkpoint_path = models_path[4]
    _, validation_geometric_arousal, test_geometric_arousal, modelo_geometric_arousal, pred_geometric_arousal, pred_validation_geometric_arousal = models.load_model_ds(np.array(data["arousal_geometric_feature"]), data["arousal_label"], "lstm", seq_len, checkpoint_path, -1, texto="Appearance features (PCA from 50k LGBP-TOP) - 168 features\n", return_seq=return_sequences, to_summary=True, to_predict=True, new_features=False)

    ds_models = {
        "modelo_audio_arousal": modelo_audio_arousal,
        "modelo_ecg_arousal": modelo_ecg_arousal,
        "modelo_hrv_arousal": modelo_hrv_arousal,
        "modelo_video_arousal": modelo_video_arousal,
        "modelo_geometric_arousal": modelo_geometric_arousal
    }

    #salva informações para próximas execuções - predições dos modelos e dados já divididos em treino validação e teste
    if not saved_information:

        #GOLD_STANDARD
        X_train, X_validation, X_test, y_train, y_validation, y_test = utils.prepare_datasets(np.array(data["arousal_audio_feature"]), data["arousal_label"], 3, 3, 1)

        _, y_validation = utils.lstm_data_transform(X_validation, y_validation, seq_len, to_shuffle=False, return_sequences=return_sequences)

        _, y_test = utils.lstm_data_transform(X_test, y_test, seq_len, to_shuffle=False, return_sequences=return_sequences)

        del y_train

        del X_train, X_validation, X_test

        bases = {
            "y_train":[], "y_validation":[], "y_test":[],
            "train_audio_arousal":[], "validation_audio_arousal":[], "test_audio_arousal":[],
            "train_ecg_arousal":[], "validation_ecg_arousal":[], "test_ecg_arousal":[], 
            "train_hrv_arousal":[], "validation_hrv_arousal":[], "test_hrv_arousal":[],
            "train_video_arousal":[], "validation_video_arousal":[], "test_video_arousal":[],
            "train_geometric_arousal":[], "validation_geometric_arousal":[], "test_geometric_arousal":[]
        }

        #bases['y_train']=y_train
        bases['y_validation']=y_validation
        bases['y_test']=y_test
        #bases['train_audio_arousal']=train_audio_arousal
        bases['validation_audio_arousal']=validation_audio_arousal
        bases['test_audio_arousal']=test_audio_arousal
        #bases['train_ecg_arousal']=train_ecg_arousal
        bases['validation_ecg_arousal']=validation_ecg_arousal
        bases['test_ecg_arousal']=test_ecg_arousal
        #bases['train_hrv_arousal']=train_hrv_arousal
        bases['validation_hrv_arousal']=validation_hrv_arousal
        bases['test_hrv_arousal']=test_hrv_arousal
        #bases['train_video_arousal']=train_video_arousal
        bases['validation_video_arousal']=validation_video_arousal
        bases['test_video_arousal']=test_video_arousal
        #bases['train_geometric_arousal']=train_geometric_arousal
        bases['validation_geometric_arousal']=validation_geometric_arousal
        bases['test_geometric_arousal']=test_geometric_arousal

        # Save data
        pickle.dump(bases, open(pickle_path, 'wb'))

        #print(bases['y_validation'].shape, bases['y_test'].shape, bases['test_audio_arousal'].shape, bases['test_ecg_arousal'].shape, bases['test_hrv_arousal'].shape, bases['test_video_arousal'].shape, bases['test_geometric_arousal'].shape)

        preds = {
            "modelo_audio_arousal": pred_audio_arousal,
            "modelo_ecg_arousal": pred_ecg_arousal,
            "modelo_hrv_arousal": pred_hrv_arousal,
            "modelo_video_arousal": pred_video_arousal,
            "modelo_geometric_arousal": pred_geometric_arousal
        }

        # Save data
        pickle.dump(preds, open(test_pred_path, 'wb'))

        validation_preds = {
            "modelo_audio_arousal": pred_validation_audio_arousal,
            "modelo_ecg_arousal": pred_validation_ecg_arousal,
            "modelo_hrv_arousal": pred_validation_hrv_arousal,
            "modelo_video_arousal": pred_validation_video_arousal,
            "modelo_geometric_arousal": pred_validation_geometric_arousal
        }

        # Save data
        pickle.dump(validation_preds, open(validation_pred_path, 'wb'))

        #competence_dist(bases, dist_path)

    else:

        bases = utils.load_data(pickle_path)
        preds = utils.load_data(test_pred_path)
        validation_preds = utils.load_data(validation_pred_path)
        final_dist = utils.load_data(dist_path)

    return ds_models, bases, preds, validation_preds, final_dist


def competence_dist(bases, dist_path):

    t_inicial = datetime.now()

    dist_audio, dist_ecg, dist_hrv, dist_video, dist_geometric = [], [], [], [], []

    test_audio_arousal = bases['test_audio_arousal']
    test_ecg_arousal = bases['test_ecg_arousal']
    test_hrv_arousal = bases['test_hrv_arousal']
    test_video_arousal = bases['test_video_arousal']
    test_geometric_arousal = bases['test_geometric_arousal']

    validation_audio_arousal = bases['validation_audio_arousal']
    validation_ecg_arousal = bases['validation_ecg_arousal']
    validation_hrv_arousal = bases['validation_hrv_arousal']
    validation_video_arousal = bases['validation_video_arousal']
    validation_geometric_arousal = bases['validation_geometric_arousal']

    for t, test in enumerate(bases['y_test']):

        print('Caso de teste: {}'.format(t+1))

        features_audio = np.array(list(chain.from_iterable(test_audio_arousal[t]))) 
        features_audio = np.array(list(chain.from_iterable(features_audio))) #features audio

        features_ecg = np.array(list(chain.from_iterable(test_ecg_arousal[t]))) 
        features_ecg = np.array(list(chain.from_iterable(features_ecg))) #features ecg

        features_hrv = np.array(list(chain.from_iterable(test_hrv_arousal[t]))) 
        features_hrv = np.array(list(chain.from_iterable(features_hrv))) #features hrv

        features_video = np.array(list(chain.from_iterable(test_video_arousal[t]))) 
        features_video = np.array(list(chain.from_iterable(features_video))) #features video

        features_geometric = np.array(list(chain.from_iterable(test_geometric_arousal[t]))) 
        features_geometric = np.array(list(chain.from_iterable(features_geometric))) #features geometric

        #print(features_audio.shape, features_ecg.shape, features_hrv.shape, features_video.shape, features_geometric.shape)

        dist_audio_v, dist_ecg_v, dist_hrv_v, dist_video_v, dist_geometric_v = [], [], [], [], []

        for v, item in enumerate(bases['y_validation']):

            features_audio_v = np.array(list(chain.from_iterable(validation_audio_arousal[v])))
            features_audio_v = np.array(list(chain.from_iterable(features_audio_v))) #features audio

            features_ecg_v = np.array(list(chain.from_iterable(validation_ecg_arousal[v]))) 
            features_ecg_v = np.array(list(chain.from_iterable(features_ecg_v))) #features ecg

            features_hrv_v = np.array(list(chain.from_iterable(validation_hrv_arousal[v]))) 
            features_hrv_v = np.array(list(chain.from_iterable(features_hrv_v))) #features hrv

            features_video_v = np.array(list(chain.from_iterable(validation_video_arousal[v]))) 
            features_video_v = np.array(list(chain.from_iterable(features_video_v))) #features video

            features_geometric_v = np.array(list(chain.from_iterable(validation_geometric_arousal[v]))) 
            features_geometric_v = np.array(list(chain.from_iterable(features_geometric_v))) #features geometric

            dist_audio_v.append(distance.euclidean(features_audio, features_audio_v))
            dist_ecg_v.append(distance.euclidean(features_ecg, features_ecg_v))
            dist_hrv_v.append(distance.euclidean(features_hrv, features_hrv_v))
            dist_video_v.append(distance.euclidean(features_video, features_video_v))
            dist_geometric_v.append(distance.euclidean(features_geometric, features_geometric_v))

            #print(features_audio_v.shape, features_ecg_v.shape, features_hrv_v.shape)

        dist_audio.append(dist_audio_v)
        dist_ecg.append(dist_ecg_v)
        dist_hrv.append(dist_hrv_v)
        dist_video.append(dist_video_v)
        dist_geometric.append(dist_geometric_v)
        print(datetime.now() - t_inicial)

    final_dist = {
        "dist_audio": [],
        "dist_ecg": [],
        "dist_hrv": [],
        "dist_video": [],
        "dist_geometric": []
    }

    final_dist['dist_audio'] = dist_audio
    final_dist['dist_ecg'] = dist_ecg
    final_dist['dist_hrv'] = dist_hrv
    final_dist['dist_video'] = dist_video
    final_dist['dist_geometric'] = dist_geometric

    # Save data
    pickle.dump(final_dist, open(dist_path, 'wb'))
