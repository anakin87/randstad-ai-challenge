import pandas as pd
from joblib import dump, load
from utils import evaluate, truncate_input, divide_words, correzioni
import time

__author__     = 'Stefano Fiorucci'
__email__      = 'stefano.fiorucci@virgilio.it'
__date__       = '2021.09.10'

import warnings
warnings.filterwarnings('ignore') 

if __name__ == '__main__':
    print('--- PROVO IL MIGLIORE MODELLO ---')
    
    best_model = load('./output/best_model.joblib')

    annuncio = 'Cerco disperatamente programmatori Spring boot a Parma'
    print('Annuncio: ' + annuncio)
    pred_start = time.time()    
    # il modello si aspetta di lavorare su una serie,
    # quindi converto l'annuncio in una serie di un solo elemento
    annuncio_as_series = pd.Series([annuncio])
    prediction = best_model.predict (annuncio_as_series)
    pred_end = time.time()    
    print(f'Predizione: {prediction[0]}')
    print(f'Tempo di predizione: {pred_end - pred_start:.2f}s')
    
    
    annuncio = 'Valutiamo l\'inserimento di un esperto in docker e virtualizzazione'
    print('Annuncio: ' + annuncio)
    pred_start = time.time()    
    annuncio_as_series = pd.Series([annuncio])
    prediction = best_model.predict (annuncio_as_series)
    pred_end = time.time()    
    print(f'Predizione: {prediction[0]}')
    print(f'Tempo di predizione: {pred_end - pred_start:.2f}s')
