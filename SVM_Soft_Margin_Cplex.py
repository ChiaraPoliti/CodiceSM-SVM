import numpy as np
import pandas as pd
from pulp import *
from typing import List, Tuple
from numpy.typing import NDArray

def SVM_Soft_Margin_Cplex(C: int, dimensions: List, observations: List, X: NDArray[np.float64], y: NDArray[np.int_]):
    """Funzione per calcolare l'iperpiano ottimo con margine geometrico massimo con Soft-Margin SVM con solver Cplex.
    
    Args:
        C (int): Parametro di penalità, definisce il peso assegnato alla classificazione esatta dei punti del dataset;
        dimensions (List): dimensione dell'iperpiano;
        observations (List): è numero di osservazioni presenti nel dataset;
        X (NDArray[np.float64]): sono le osservazioni del dataset;
        y (NDArray[np.int_]): sono le etichette delle osservazioni del dataset.

    Returns:
        a (NDArray[np.float64]): nell'equazione di un iperpiano ax=b rappresenta a, vettore riga di dimensione n
        b (float): nell'equazione di un iperpiano ax=b rappresenta b;
        z (Dict[int, float]): esito delle classificazioni dei dati: uguale a 0 se il punto si trova nello spazio corretto, compreso tra 0 e 1 se si trova nello spazio compreso tra i maqrgini superiore e inferiore, pari a 1 se si trova nello spazio errato;
    """
    #Modello Pulp per SM-SVM 
    Soft_Margin_SVM_model = LpProblem("Soft_Margin_SVM_model", LpMinimize)

    #Variabili
    a = LpVariable.dicts('a', (dimensions))
    b = LpVariable("b")
    z = LpVariable.dicts('z', (observations), lowBound=0)

    #Variabile ausiliaria per norma 1
    u = LpVariable.dicts('u', (dimensions))

    #Funzione obiettivo: massimizzare il margine geometrico e minimizzare il numero di misclassificazioni
    Soft_Margin_SVM_model += 0.5*lpSum(u[d] for d in dimensions) + C*lpSum(z[i] for i in observations)

    # Vincoli di linearizzazione della norma L1
    for d in dimensions:
        Soft_Margin_SVM_model += u[d] >= a[d], f'Abs_constraint_1_for_dimension_{d}'
        Soft_Margin_SVM_model += u[d] >= -a[d], f'Abs_constraint_2_for_dimension_{d}'

    #Vincoli di separazione
    for i in observations:
        Soft_Margin_SVM_model += y[i] * (lpSum(a[d] * X[i, d] for d in dimensions) - b) >= 1 - z[i], f'Classify_observation_{i}'

    # Risoluzione del modello
    Soft_Margin_SVM_model.solve(solver=CPLEX())

    a_opt = [value(a[d]) for d in dimensions]
    b_opt = value(b)

    return a_opt, b_opt, z