# Randstad Artificial Intelligence Challenge (powered by VGEN)
Soluzione proposta da Stefano Fiorucci (anakin87) - primo classificato

##Struttura directory del progetto
+ directory input:
    + train_set.csv e test_set.csv *(assenti per motivi di copyright, reperibili mediante iscrizione su https://www.vgen.it/randstad-artificial-intelligence-challenge/)*
    + correzioni.json: file JSON contenente le correzioni per le parole erroneamente
accorpate nel dataset
+ directory output:
    * best_model.joblib: il migliore modello addestrato (su Windows), salvato con
la libreria [joblib](https://scikit-learn.org/stable/modules/model_persistence.html)
    *  best_predictions.csv: file CSV delle predizioni del miglior modello sul test
set, contenente le colonne Job_description, Label_true e Label_pred; il
separatore è“;”*(assente per motivi di copyright)*

+ directory principale:
    + esplorazione_scelta_modello.ipynb: il notebook python che descrive il
percorso di esplorazione e scelta del migliore modello machine learning
    + esplorazione_scelta_modello.html: esportazione in formato HTML del
suddetto notebook
    + logo.jpg: logo della competizione 
    + readme.md: questa guida
	+ requirements.txt: le librerie python da installare per riprodurre l'ambiente di
addestramento/predizione
    + presentation.pdf: la presentazione della soluzione proposta
	+ train_model_windows.py: versione Windows dello script python che
consente di ripetere l'addestramento, la valutazione del modello, il
salvataggio del modello e la scrittura del CSV con le predizioni
	+ train_model_linux.py: versione Linux dello script python di addestramento
	+ utils.py: modulo python contenente alcune funzioni necessarie per il training e
la predizione
	+ try_best_model.py: script python di esempio che mostra come caricare il
modello salvato e usarlo per nuove predizioni

## Preparazione dell'ambiente di esecuzione

Per eseguire gli script, è necessario Python>=3.6.
Si consiglia di preparare l’ambiente di esecuzione mediante i seguenti passaggi:
1. scaricamento del repository
2. a partire dalla directory principale, creazione di un python virtual environment con il
comando
`python3 -m venv venv`
3. attivazione del virtual environment
a. windows
`venv\Scripts\activate`
b. linux
`source venv/bin/activate`
4. installazione delle librerie necessarie con il comando
`pip install -r requirements.txt`

## Esecuzione degli script

- *try_best_model* è uno script python di esempio che mostra come caricare il migliore
modello salvato e usarlo per nuove predizioni
si lancia con la sintassi
`python try_best_model.py`
-  Lo script *train_model* lancia l’addestramento del modello, seguito dalla stampa delle metriche valutate sul test set e può essere eseguito con la sintassi
  - Windows
`python train_model_windows.py`
  - Linux
`python train_model_linux.py`
Possono essere specificati i parametri:
--save-model (oppure -s), che salva il modello appena addestrato nella directory
output, con un nome file indicante data e ora
--get-predictions (oppure -p), che genera le predizioni sul test set in formato csv e le
salva nella directory di output, con un nome file indicante data e ora

### Nota
A causa di un bug noto di [numpy](https://github.com/numpy/numpy/issues/11500), l'addestramento dei modelli su Windows e Linux non è completamente identico e, a parità di parametri e random state, produce
modelli leggermenti diversi, con effetti sulle performance (F1).

Si è cercato il più possibile di ottenere modelli con performance vicine nei due sistemi operativi (facendo variare il random state).

**Il migliore modello è stato addestrato in ambiente Windows** ed è salvato come
best_model.joblib. Le predizioni migliori (best_predictions.csv) sono relative a questo
modello. Usando lo script fornito (train_model_windows.py), il modello può essere
riaddestrato rapidamente (pochi secondi) in ambiente Windows. Anche se addestrato
su Windows, può essere correttamente impiegato su Linux per la predizione.

Il modello per Linux, addestrabile con l’apposito script (train_model_linux.py), è molto
simile a quello per Windows: le differenze riscontrabili a livello di performance (F1)
sono inferiori a 0.001.

*Attenzione:* usando lo script di addestramento per Windows in ambiente Linux o
viceversa, non si ottengono errori di esecuzione, ma il modello addestrato mostra
delle performance qualitative (F1) inferiori a quelle attese.