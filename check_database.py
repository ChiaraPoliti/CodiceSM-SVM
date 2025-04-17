import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from pulp import *
from pulp import GUROBI
from typing import List, Tuple
from numpy.typing import NDArray
#from sklearn.metrics import confusion_matrix
#import time

def check_heart ():
    """Funzione che restituisce il database Heart Failure pronto all'uso"""

    #Carico il file dati: è un dataset sugli infarti cardiaci
    df_heart_failure = pd.read_csv("heart_failure_clinical_records_dataset.csv")

    #Conversione delle etichette: '1' -> 1, '0' -> -1
    df_heart_failure['DEATH_EVENT'] = df_heart_failure['DEATH_EVENT'].map({1: 1, 0: -1})

    return df_heart_failure

def check_kidney ():
    """Funzione che restituisce il database Chronic Kidney Disease pronto all'uso """

    #Carico il dataset: tratta disfunzioni renali
    df_kidney = pd.read_csv("chronic_kidney_disease_full.csv", header=None)

    #Sostituisco i valori sconosciuti con NaN
    df_kidney = df_kidney.replace(['?', 'None', ''], np.nan)

    #Converto il tipo di dato per permettere di calcolare la media
    cols_da_convertire = [0,1,2,3,4,9,10,11,12,13,14,15,16,17]
    for col in cols_da_convertire:
        df_kidney[col] = pd.to_numeric(df_kidney[col], errors='coerce')

    #Gestisco i NaN numerici con la media
    for col in df_kidney.select_dtypes(include=np.number).columns:
        df_kidney[col].fillna(df_kidney[col].mean(), inplace=True)

    #Converto anche i valori binari in bit numerici
    df_kidney[5] = df_kidney[5].map({'normal': 1, 'abnormal': -1})
    df_kidney[6] = df_kidney[6].map({'normal': 1, 'abnormal': -1})
    df_kidney[7] = df_kidney[7].map({'present': 1, 'notpresent': -1})
    df_kidney[8] = df_kidney[8].map({'present': 1, 'notpresent': -1})
    df_kidney[18] = df_kidney[18].map({'yes': 1, 'no': -1})
    df_kidney[19] = df_kidney[19].map({'yes': 1, 'no': -1})
    df_kidney[20] = df_kidney[20].map({'yes': 1, 'no': -1})
    df_kidney[21] = df_kidney[21].map({'good': 1, 'poor': -1})
    df_kidney[22] = df_kidney[22].map({'yes': 1, 'no': -1})
    df_kidney[23] = df_kidney[23].map({'yes': 1, 'no': -1})
    df_kidney[24] = df_kidney[24].map({'ckd': 1, 'notckd': -1})

    #Gestisco i valori mancanti delle colonne binarie con la moda, ricordando che mi produce una Series e devo prendere solo il primo valore per non avere problemi
    cols_binarie = [5,6,7,8,18,19,20,21,22,23,24]
    for col in cols_binarie:
        moda = df_kidney[col].mode()
        if not moda.empty:
            valore_moda = moda.iloc[0]
            df_kidney[col].fillna(valore_moda, inplace=True)

    return df_kidney

def check_rice ():
    """Funzione che restituisce il database Rice Classification pronto all'uso """

    #Carico il dataset: tratta di una classificazione del riso 
    df_rice = pd.read_csv("Rice_Cammeo_Osmancik.csv", header = None)

    #Conversione delle etichette: 'Cammeo' -> 1, 'Osmancik' -> -1
    df_rice[7] = df_rice[7].map({'Cammeo': 1, 'Osmancik': -1})

    return df_rice

def check_breast ():
    """Funzione che restituisce il database Breast Cancer pronto all'uso """

    #Carico il dataset: tratta di tumori benigni/maligni al seno
    df_breast = pd.read_csv("wdbc.csv", header = None)

    #Elimino la colonna id
    df_breast.drop(columns = [0], inplace=True)

    #Converto le etichette del target
    df_breast[1] = df_breast[1].map({'B': 1, 'M': -1})

    #Sposto la colonna target in ultima posizione
    nome_prima_colonna = df_breast.columns[0]
    prima_colonna = df_breast.pop(nome_prima_colonna)
    ultima_posizione = len(df_breast.columns)
    df_breast.insert(ultima_posizione, nome_prima_colonna, prima_colonna)

    return df_breast

def check_spam ():
    """Funzione che restituisce il database Spam Mail pronto all'uso """

    #Carico il dataset: tratta di classificazione delle e-mail in spam o non spam
    df_spam = pd.read_csv("spambase.csv", header = None)

    #Converto le etichette del target
    df_spam[57] = df_spam[57].map({1: 1, 0: -1})

    return df_spam

def check_eyes ():
    """Funzione che restituisce il database Eyes pronto all'uso """

    #Carico il dataset: classifica gli occhi come aperti o chiusi
    df_eyes = pd.read_csv("EEG_Eye_State.csv", header = None)

    #Converto le etichette del target
    df_eyes[14] = df_eyes[14].map({1: 1, 0: -1})

    return df_eyes

def check_blood ():
    """Funzione che restituisce il database Blood Transfusion pronto all'uso """

    #Carico il dataset: tratta di trasfusioni di sangue
    df_blood = pd.read_csv("transfusion.csv")

    #Converto le etichette del target
    df_blood['whether he/she donated blood in March 2007'] = df_blood['whether he/she donated blood in March 2007'].map({1: 1, 0: -1})

    return df_blood

def check_fert ():
    """Funzione che restituisce il database Fertility pronto all'uso """

    #Carico il dataset: tratta di problemi di fertilità
    df_fert = pd.read_csv("fertility_Diagnosis.csv", header = None)

    #Converto le etichette del target
    df_fert[9] = df_fert[9].map({'N': 1, 'O': -1})

    return df_fert