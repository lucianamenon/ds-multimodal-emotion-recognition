"""
Preprocess audio, video and physiological data and saves segmented information
"""

# AUDIO FEATURES:
# Acoustic features openSMILE and eGeMAPS - features_audio/*.arff (88 features)

# PHYSIOLOGICAL FEATURES:
# Electrocardiogram (ECG)- features_ECG/*.arff (19 features)

# VIDEO FEATURES:
# PCA from 50k LGBP-TOP features (LGBP-TOP) - features_video_appearance/*.arff (168 features)
# features derived from 49 facial landmarks (FACIAL LANDMARKS) - features_video_geometric/*.arff (632 features)

import os
import pickle
import time
import arff

from tqdm import tqdm
import numpy as np

import config

RECOLA_DIR = config.RECOLA_DIR
RECOLA_VIDEO_DIR = config.RECOLA_VIDEO_DIR
RECOLA_PICKLE_PATH = config.RECOLA_PICKLE_PATH
GS_DIR = config.GS_DIR

def get_samples(subject_id):
    """Recupera audio, video and physiological data e o label de cada subsample de 40ms da base de dados
    segment (float): Time in seconds to segment the waveform"""

    # GOLD STANDARD
    # ratings_gold_standard/*.arff
    # arff files containing gold standard ratings (will be used to evaluate performance on test partition) for arousal and valence for training and development sets
    # 1 feature x 7501 frames per file, values were obtained by averaging over the 6 raters
    arousal_label_path = GS_DIR / 'arousal/{}.arff'.format(subject_id)
    valence_label_path = GS_DIR / 'valence/{}.arff'.format(subject_id)

    # load gold standard file
    arousal_label = np.array(arff.load(open(arousal_label_path, 'r'))['data'])
    valence_label = np.array(arff.load(open(valence_label_path, 'r'))['data'])

    # [:, 2][1:] => [:, 2] Recupera apenas a última coluna que é a informação desejada (label)
    # [:, 2][1:] => [1:] Remove o header (não está relacionado a nenhum segmento de audio)
    arousal_label = arousal_label[:, 2].astype(float)
    valence_label = valence_label[:, 2].astype(float)

    # FEATURES AUDIO
    # Acoustic features openSMILE and eGeMAPS - features_audio/*.arff (88 features)
    # features_audio/*.arff
    # arff files containing acoustic features computed with openSMILE and the eGeMAPS configuration file, with a sliding centred window which size depends on the modality (arousal -> ws=4s, valence -> ws=6s; optimised on the dev partition).
    # Features are thus provided separately for each of those two dimensions.
    # The first feature vector is assigned to the center of the window, and duplicated for the previous frames - from 1 to ws/(2*sampling_period)-1, with the sampling period being equal to 40ms.
    # The last feature vector is also duplicated for the last frames.
    # 88 features x 7501 frames per file
    arousal_audio_features_path = RECOLA_DIR / \
        'features_audio/arousal/{}.arff'.format(subject_id)
    valence_audio_features_path = RECOLA_DIR / \
        'features_audio/valence/{}.arff'.format(subject_id)

    # load audio features file
    arousal_audio_feature = np.array(
        arff.load(open(arousal_audio_features_path, 'r'))['data'])
    valence_audio_feature = np.array(
        arff.load(open(valence_audio_features_path, 'r'))['data'])
    arousal_audio_feature = arousal_audio_feature[:, 1:].astype(float)
    valence_audio_feature = valence_audio_feature[:, 1:].astype(float)

    # FEATURES VIDEO
    # features_video_appearance/*.arff
    # arff files containing appearance video features obtained by a PCA from 50k LGBP-TOP features (99% of variance)
    # Piecewise cubic Hermit interpolation over time is performed according to "recordings_video_frame_time"
    # Values are forced to 0 on frames for which the face was not detected.
    # Arithmetic mean and standard deviation are then computed on all descriptors with a sliding centred window which size depends on the modality (arousal -> ws=6s, valence -> ws=4s; optimised on the dev partition).
    # Features are thus provided separately for each of those two dimensions.
    # The first feature vector is assigned to the center of the window, and duplicated for the previous frames - from 1 to ws/(2*sampling_period)-1, with the sampling period being equal to 40ms.
    # The last feature vector is also duplicated for the last frames.
    # 168 features x 7501 frames per file
    arousal_video_features_path = RECOLA_DIR / \
        'features_video_appearance/arousal/{}.arff'.format(subject_id)
    valence_video_features_path = RECOLA_DIR / \
        'features_video_appearance/valence/{}.arff'.format(subject_id)

    # load audio features file
    arousal_video_feature = np.array(
        arff.load(open(arousal_video_features_path, 'r'))['data'])
    valence_video_feature = np.array(
        arff.load(open(valence_video_features_path, 'r'))['data'])
    arousal_video_feature = arousal_video_feature[:, 1:].astype(float)
    valence_video_feature = valence_video_feature[:, 1:].astype(float)

    # features_video_geometric/*.arff
    # arff files containing geometric video features derived from 49 facial landmarks
    # Piecewise cubic Hermit interpolation over time is performed according to "recordings_video_frame_time"
    # Values are forced to 0 on frames for which the face was not detected.
    # Arithmetic mean and standard deviation are then computed on all descriptors with a sliding centred window which size depends on the modality (arousal -> ws=4s, valence -> ws=8s; optimised on the dev partition).
    # Features are thus provided separately for each of those two dimensions.
    # The first feature vector is assigned to the center of the window, and duplicated for the previous frames - from 1 to ws/(2*sampling_period)-1, with the sampling period being equal to 40ms.
    # The last feature vector is also duplicated for the last frames.
    # 632 features x 7501 frames per file
    arousal_geometric_features_path = RECOLA_DIR / \
        'features_video_geometric/arousal/{}.arff'.format(subject_id)
    valence_geometric_features_path = RECOLA_DIR / \
        'features_video_geometric/valence/{}.arff'.format(subject_id)

    # load audio features file
    arousal_geometric_feature = np.array(
        arff.load(open(arousal_geometric_features_path, 'r'))['data'])
    valence_geometric_feature = np.array(
        arff.load(open(valence_geometric_features_path, 'r'))['data'])
    arousal_geometric_feature = arousal_geometric_feature[:, 1:].astype(float)
    valence_geometric_feature = valence_geometric_feature[:, 1:].astype(float)

    #FEATURES BATIMENTO CARDÍACO
    #Eletrocardiograma (ECG)- features_ECG/*.arff (19 features)
    arousal_ecg_path = RECOLA_DIR / 'features_ECG/arousal/{}.arff'.format(subject_id)
    valence_ecg_path = RECOLA_DIR / 'features_ECG/valence/{}.arff'.format(subject_id)
    arousal_ecg = np.array(
        arff.load(open(arousal_ecg_path, 'r'))['data'])
    valence_ecg = np.array(
        arff.load(open(valence_ecg_path, 'r'))['data'])
    arousal_ecg = arousal_ecg[:, 1:].astype(float)
    valence_ecg = valence_ecg[:, 1:].astype(float)

    #Heart Rate First Order Derivate (HRV) - features_HRHRV/*.arff (10 features)
    arousal_hrv_path = RECOLA_DIR / 'features_HRHRV/arousal/{}.arff'.format(subject_id)
    valence_hrv_path = RECOLA_DIR / 'features_HRHRV/valence/{}.arff'.format(subject_id)
    arousal_hrv = np.array(arff.load(open(arousal_hrv_path, 'r'))['data'])
    valence_hrv = np.array(arff.load(open(valence_hrv_path, 'r'))['data'])
    arousal_hrv = arousal_hrv[:, 1:].astype(float)
    valence_hrv = valence_hrv[:, 1:].astype(float)

    return arousal_audio_feature, valence_audio_feature, arousal_ecg, valence_ecg, arousal_hrv, valence_hrv, arousal_video_feature, valence_video_feature, arousal_geometric_feature, valence_geometric_feature, arousal_label, valence_label, np.dstack([arousal_label, valence_label])[0].astype(np.float32)


def serialize_sample_json(data, subject_id, debug=True):

    for i, (arousal_audio_feature, valence_audio_feature, arousal_ecg, valence_ecg, arousal_hrv, valence_hrv, arousal_video_feature, valence_video_feature, arousal_geometric_feature, valence_geometric_feature, arousal_label, valence_label, labels) in enumerate(zip(*get_samples(subject_id))):
        data["arousal_audio_feature"].append(arousal_audio_feature)
        data["valence_audio_feature"].append(valence_audio_feature)
        data["arousal_video_feature"].append(arousal_video_feature)
        data["valence_video_feature"].append(valence_video_feature)
        data["arousal_geometric_feature"].append(arousal_geometric_feature)
        data["valence_geometric_feature"].append(valence_geometric_feature)
        data["arousal_ecg"].append(arousal_ecg)
        data["valence_ecg"].append(valence_ecg)
        data["arousal_hrv"].append(arousal_hrv)
        data["valence_hrv"].append(valence_hrv)
        data["arousal_label"].append(arousal_label)
        data["valence_label"].append(valence_label)

        if debug:
            print(f"{subject_id}, segment: {i+1}")

        del arousal_audio_feature, valence_audio_feature, arousal_ecg, valence_ecg, arousal_hrv, valence_hrv, arousal_video_feature, valence_video_feature, arousal_geometric_feature, valence_geometric_feature, arousal_label, valence_label


def data_generation():

    print('Starting preprocessing audio, video and physiological data and saving segmented information...')

    t_inicio = time.time()

    # Dictionary to store subsampled data and labels (40 ms annotation frequency)
    data = {
        # "video_images": [],
        # "video_images_gray": [],
        # "raw_audio": [],
        # "mfccs": [],
        # "specs": [],
        "arousal_video_feature": [],
        "valence_video_feature": [],
        "arousal_geometric_feature": [],
        "valence_geometric_feature": [],
        "arousal_audio_feature": [],
        "valence_audio_feature": [],
        "arousal_ecg": [],
        "valence_ecg": [],
        "arousal_hrv": [],
        "valence_hrv": [],
        "arousal_label": [],
        "valence_label": []
    }

    for subj in tqdm(sorted(os.listdir(RECOLA_VIDEO_DIR))):
        subj_id = subj.split('.')[0]
        portion = subj.split('_')[0]
        if portion != 'test':
            serialize_sample_json(data, subj_id)

    t2 = time.time()
    print("Tempo total: {}".format(t2-t_inicio))

    # Save data
    pickle.dump(data, open(RECOLA_PICKLE_PATH, 'wb'))
    print("Preprocessed data path: {}".format(RECOLA_PICKLE_PATH))
