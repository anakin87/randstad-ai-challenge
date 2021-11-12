import pandas as pd
import json
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif
from joblib import dump, load
import optparse
import datetime
import time
from utils import evaluate, truncate_input, divide_words

__author__     = 'Stefano Fiorucci'
__email__      = 'stefano.fiorucci@virgilio.it'
__date__       = '2021.09.10'


import warnings
warnings.filterwarnings('ignore') 

if __name__ == '__main__':
    parser = optparse.OptionParser(
        description='Addestra il modello machine learning e mostra le metriche. '
        'Se specificato, può salvare il modello (--save-model) '
        'e produrre il CSV con le predizioni sul test set (--get-predictions)')
    parser.add_option('-s', '--save-model', action='store_true',
                      dest='save_model', default=False,
                      help='salva il modello nella directory models')
    parser.add_option('-p', '--get-predictions', action='store_true',
                      dest='get_predictions', default=False,
                      help='genera le predizioni sul test set in formato CSV') 
    (options, args) = parser.parse_args()
    
    # leggo training set e test set
    df_train=pd.read_csv('./input/train_set.csv')
    df_test=pd.read_csv('./input/test_set.csv')


        
    print('Addestramento modello...')
    training_start = time.time()    
    best_pipe = Pipeline([
            ('divide_words', FunctionTransformer(func=divide_words)),
            ('trunc_input', FunctionTransformer(func=truncate_input,
                                                kw_args={'n_words':600})),
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_df=0.7)),
            ('select_features', SelectPercentile(score_func=f_classif, 
                                                 percentile=96.97)),
            ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                alpha=1e-4, max_iter=11, 
                                tol=1e-4, early_stopping=False,
                                random_state=22280)),
    ])
    best_pipe.fit(df_train.Job_offer, df_train.Label)
    training_end = time.time()   
    print(f'Tempo di addestramento: {training_end - training_start:.2f}s')
        
    evaluate(best_pipe, df_test)
    
    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")  
    # salvataggio modello (con joblib)
    if options.save_model:
        model_filepath = f'./output/model_{timestamp}.joblib'
        print(f'Salvataggio modello in {model_filepath}')
        dump(best_pipe, model_filepath)
    
    # scrittura file csv delle predizioni sul test        
    if options.get_predictions:
        predictions = df_test.rename({'Job_offer':'Job_description',
                    'Label':'Label_true'}, axis='columns')
        predictions['Label_pred'] = best_pipe.predict(df_test.Job_offer)
        
        predictions_filepath = f'./output/predictions_{timestamp}.csv'
        print(f'Salvataggio predizioni in {predictions_filepath}')
        predictions.to_csv(predictions_filepath, sep=';', index=False)