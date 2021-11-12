import pandas as pd
import json
from sklearn import metrics
import optparse
import datetime

__author__     = 'Stefano Fiorucci'
__email__      = 'stefano.fiorucci@virgilio.it'
__date__       = '2021.09.10'


import warnings
warnings.filterwarnings('ignore') 

def evaluate(model, df_test):
    """
    Funzione per stampare le metriche di interesse sul dataset di test,
    in base al modello scelto
    """
    
    test_pred = model.predict(df_test.Job_offer)
    
    print('Valutazione del modello sul test set:\n',
          metrics.classification_report(
        df_test.Label, test_pred , digits=5))

    precision_macro = metrics.precision_score(df_test.Label, test_pred, average='macro')
    precision_micro = metrics.precision_score(df_test.Label, test_pred, average='micro')
    precision_weighted = metrics.precision_score(df_test.Label, test_pred, average='weighted')
    precision_string=f"""precision micro: {precision_micro:.5f}
precision macro: {precision_macro:.5f}
precision weighted: {precision_weighted:.5f}"""
    print(precision_string)

    recall_macro = metrics.recall_score(df_test.Label, test_pred, average='macro')
    recall_micro = metrics.recall_score(df_test.Label, test_pred, average='micro')
    recall_weighted = metrics.recall_score(df_test.Label, test_pred, average='weighted')
    recall_string=f"""recall micro: {recall_micro:.5f}
recall macro: {recall_macro:.5f}
recall weighted: {recall_weighted:.5f}"""
    print(recall_string)              
    
    f1_macro = metrics.f1_score(df_test.Label, test_pred, average='macro')
    f1_micro = metrics.f1_score(df_test.Label, test_pred, average='micro')
    f1_weighted = metrics.f1_score(df_test.Label, test_pred, average='weighted')
    f1_string=f"""f1 micro: {f1_micro:.5f}
f1 macro: {f1_macro:.5f}
f1 weighted: {f1_weighted:.5f}"""
    print(f1_string)
    
   

def divide_words(series):
    """
    Funzione che, applicata alla serie,
    per ogni elemento divide le parole accorpate
    """
    new_sentences=[]
    
    lower_series=series.str.lower()
    for sentence in lower_series:
        new_sentence=sentence
        for composite_word,words in correzioni.items():
            if composite_word in new_sentence:
                new_sentence=new_sentence.replace(composite_word,words)
        new_sentences.append(new_sentence)
    new_series=pd.Series(new_sentences)
    return new_series

def truncate_input(series,n_words):
    """
    Funzione che, applicata alla serie,
    per ogni elemento elimina le parole che eccedono n_words
    """
    splitted_series=series.str.split()
    truncated_series=pd.Series([' '.join(word[:n_words])
                                for word in splitted_series])
    return truncated_series

# carico il dizionario delle correzioni per parole accorpate
with open('./input/correzioni.json') as fin:
    correzioni=json.load(fin)
