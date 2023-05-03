
# MULTIMODAL EMOTION RECOGNITION: A Dynamic Selection Based Approach


## About The Project
Our experience of the world is multimodal - we see objects, hear sounds, feel the textures, smell odors, and taste flavors (Baltrusaitis; Ahuja e Morency, 2019). The main objective of this research project is to explore and evaluate different ways of representing data and define the best modality, view, or set of features for each situation. The aim of the research is focused on the **dynamic selection** of the best data representation according to the test example.

### Important notes:
All experiments were carried out with pre-processed features available in the RECOLA: Remote Collaborative and Affective Interactions (AVEC_2016 dataset).

### Proposed method:

!images/metodo.png

A partir de múltiplos modais (áudio, eletrocardiograma e vídeo) e múltiplas views desses modais, numa primeira fase nós temos a geração de um pool de regressores f (todos os regressores são treinados separadamente).

Durante o teste, ocorre a segunda fase, que é a seleção dinâmica em si. Nessa fase os nossos modelos são avaliados e cada um deles recebe um peso para avaliar cada caso de teste, de acordo com sua assertividade na zona de competência (região similar ao caso de teste).
- Temos um problema de regressão, multimodal-multiview e que leva em conta recortes temporais dos dados (devido ao uso de LSTMs, apesar da base ser rotulada de 40 em 40ms, a entrada na LSTM depende de um recorte temporal maior e nós trabalhamos com janelas de 2s, seq_len=50).

Para cada caso de teste (considerem aqui um caso de teste como uma janela de 2s) nós encontramos a região de competência (K vizinhos mais similares a este caso de teste na base de validação). Como estamos falando de um problema com múltiplos conjuntos de features, nós encontramos os K vizinhos para cada representação / para cada view. Nos últimos testes trabalhamos com k=15, então para ecg encontramos os 15 conjuntos de features de ecg mais próximos do nosso ecg de teste, para egemaps encontramos os 15 egemaps mais parecidos com nosso egemaps do caso de teste e assim por diante.

-  É atribuído um peso para cada item da região de competência (levando em conta a distância entre o caso de teste e o item da região de competência).
- Na última parte do cálculo encontramos o peso dos regressores (levando em conta as distâncias com os itens da zona de competência e o erro obtido pelo regressor na zona de competência) - aqui trabalhamos com o todo, observem que na parte de cima da expressão temos distância vezes erro de um regressor (relacionado a um conjunto de features), mas isso tá ponderado com todos os N regressores.
- O resultado final é obtido pela soma do resultado de cada regressor de f multiplicado pelo seu peso.

Basicamente levamos em conta a similaridade do caso de teste com sua zona de competência e a assertividade do regressor nessa zona de competência, se os vizinhos forem próximos e o erro baixo o regressor tem um peso maior, se o vizinho for mais distante ou o erro alto o regressor tem um peso menor. 

Para fim de comparação no código serão econtrados resultados de três abordagens diferentes:
- Dynamic Selection (DS): Selecionamos apenas o regressor com menor erro acumulado na região de competência. 
- Dynamic Weighting (DW): Combina todos os regressores do ensemble conforme explicado acima.
- Dynamic Weighting with Selection (DWS): Combina um subconjunto dos regressores, estamos trabalhando com duas ideias, DWS1 e DWS2. Em DWS1 os regressores com o erro médio superior a um valor de threshold definido são descartados e em DWS2 os regressores que possuem itens da zona de competência com o erro superior a um valor de threshold são penalizados.

### Running

Clone the repo, set an python 3 virtual environment, install the dependencies from requirementes.txt and run the project with one of the options below:

```
python main.py preprocess 
```
Em *preprocess* os dados serão lidos de RECOLA_DIR, RECOLA_VIDEO_DIR e GS_DIR e salvos em RECOLA_PICKLE_PATH.

```
  python main.py train 
```
Em *train* os modelos serão treinados a partir dos dados de RECOLA_PICKLE_PATH e salvos em MODELS_DIR. Pode ser utilizado com os valores default de seq_len=50, return_sequences=False e models='all' ou personalizado de acordo com o necessário. 

  Exemplo: python main.py train  --return_sequences --models ecg

```
  python main.py ds
```
Em *ds* é realizada a seleção dinâmica da melhor representação dos dados (DS, DW, DWS1, DWS2) a partir dos modelos treinados e listados em PATH_MODELOS. Pode ser utilizado com os valores default de k=15, seq_len=50, return_sequences=False e saved_information=False ou personalizado de acordo com o necessário. 

  Exemplo: python main.py ds --return_sequences --saved_information --seq_len 150 --k 1

--saved_information é a flag que define se serão utilizados os dados de predições, train_test_split e distâncias previamete salvos ou se serão recauculados (atenção que neste caso os dados serão sobrescritos).

salvos em RECOLA_PICKLE_PATH_SPLIT_50_TRUE, RECOLA_TEST_PRED_PATH_50_TRUE, 
RECOLA_VALIDATION_PRED_PATH_50_TRUE, RECOLA_VALIDATION_DIST_PATH_50_TRUE
### Configuration file

Pode ser necessária alteração no arquivo de configuração (config/settings.py).

```
  RECOLA_DIR: Path da base de dados RECOLA AVEC_2016
  RECOLA_VIDEO_DIR: Path de 'AVEC_2016/recordings_video'
  GS_DIR: Path de 'AVEC_2016/ratings_gold_standard' 
  RECOLA_PICKLE_PATH: Path dos dados pré-processados, pode ser utilizado o que eu estou fornecendo ou gerados novamente via preprocess
  SAMPLE_RATE: Sample rate, dafault = 16000
  RECOLA_SUBJECTS: Número de pessoas da base, default = 18
  SEGMENTS_PER_PERSON: Número de segmentos rotulados por pessoa, default = 7500
  MODELS_DIR: Path onde os modelos serão salvos
  MODELOS: Títulos dos modelos, usado nos plots dos gráficos
  PATH_MODELOS_50_TRUE: Path dos modelos treinados de audio, ecg, hrv, video e geometric features (nesta ordem, com seq_len=50 e return_sequences=True)
  RECOLA_PICKLE_PATH_SPLIT_50_TRUE: Train, test, validation splits previamete salvos, utilizado para otimizar ds com --saved_information
  RECOLA_TEST_PRED_PATH_50_TRUE: Predições previamete salvas base de teste, utilizado para otimizar ds com --saved_information
  RECOLA_VALIDATION_PRED_PATH_50_TRUE: Predições previamete salvas base de validação, utilizado para otimizar ds com --saved_information
  RECOLA_VALIDATION_DIST_PATH_50_TRUE: Distâncias zona de competência previamete salvas, utilizado para otimizar ds com --saved_information

```

## Contact

* E-mail : trinkaus.luciana@gmail.com 
