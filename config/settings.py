from pathlib import Path

class baseConfig():
    RECOLA_DIR = Path("/home/luciana/Doutorado/BASES/AVEC_2016_PastaCompactada/AVEC_2016")
    RECOLA_VIDEO_DIR = RECOLA_DIR / 'recordings_video'
    GS_DIR = RECOLA_DIR / 'ratings_gold_standard'
    RECOLA_PICKLE_PATH = Path('resources/data.p')
    SAMPLE_RATE = 16000
    RECOLA_SUBJECTS = 18
    SEGMENTS_PER_PERSON = 7500
    MODELS_DIR = Path('resources/models')
    MODELOS = ['Gold Standard', 'Acoustic features (eGeMAPS)', 'Eletrocardiograma (ECG)', 'Heart Rate First Order Derivate (HRV)', 'Appearance features (PCA from 50k LGBP-TOP)', 'Geometric features (derived from 49 facial landmarks)']
    PATH_MODELOS_50_TRUE = [MODELS_DIR / 'models-20230322-030033', MODELS_DIR / 'models-20230322-031145',  MODELS_DIR / 'models-20230322-031602',  MODELS_DIR / 'models-20230318-205541',  MODELS_DIR / 'models-20230318-211604']
    RECOLA_PICKLE_PATH_SPLIT_50_TRUE = Path('resources/data_split_50_TRUE.p')
    RECOLA_TEST_PRED_PATH_50_TRUE = Path('resources/pred_TEST_50_TRUE.p')
    RECOLA_VALIDATION_PRED_PATH_50_TRUE = Path('resources/pred_VALIDATION_50_TRUE.p')
    RECOLA_VALIDATION_DIST_PATH_50_TRUE = Path('resources/dist_VALIDATION_50_TRUE.p')
    PATH_MODELOS_50_FALSE = ''
    RECOLA_PICKLE_PATH_SPLIT_50_FALSE = ''
    RECOLA_TEST_PRED_PATH_50_FALSE = ''
    RECOLA_VALIDATION_PRED_PATH_50_FALSE = ''
    RECOLA_VALIDATION_DIST_PATH_50_FALSE = ''
