import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import *
from typing import List, Tuple
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix
import time
from check_database import *
from SVM_Soft_Margin_Cplex import SVM_Soft_Margin_Cplex

#Fisso le variabili su cui farò girare il codice
number_of_tries = 100
training_percentages = np.array([0.75,0.5,0.10])
c_values = [0.5,100]
df_heart_failure = check_heart()
df_kidney = check_kidney()
df_rice = check_rice()
df_breast = check_breast()
df_spam = check_spam()
df_eyes = check_eyes()
#df_blood = check_blood()
#df_fert = check_fert()

datasets = {
    "heart_failure": df_heart_failure,
    "kidney": df_kidney,
    "rice": df_rice,
    "breast": df_breast,
    "spam":df_spam,
    "eyes": df_eyes
}


#Ciclo sul dataset, sul valore C, sulla percentuale di train e sul numero di tentativi
results = []
#accuracy = []
for dataset, df in datasets.items():
    for C in c_values:
        for training_percentage in training_percentages:
            tik = time.time()
            for try_number in range(number_of_tries):
                # Sample data
                df_training = df.sample(frac=training_percentage, random_state=try_number)
                df_testing = df.drop(df_training.index)

                #Estraggo i dati del Train ed escludo la variabile target, che per ogni dataset è l'ultima colonna
                X_train = df_training.iloc[:, :-1].values
                y_train = df_training.iloc[:, -1].values
                observations_training = range(X_train.shape[0])
                dimensions = range(X_train.shape[1])

                #Estraggo i dati del Test ed escludo la variabile target
                X_test = df_testing.iloc[:, :-1].values
                y_test = df_testing.iloc[:, -1].values
                observations_testing = range(X_test.shape[0])

                #Trovo l'iperpiano ottimo usando la funzione definita in precedenza
                a, b, z = SVM_Soft_Margin_Cplex(C, dimensions, observations_training, X_train, y_train)

                #Recupero i valori predetti sul test
                y_pred_test = []
                for x in X_test:
                    prediction = np.dot(a,x) - b
                    if prediction > 0:
                        y_pred_test.append(1)
                    else:
                        y_pred_test.append(-1)

                #Metriche di valutazione della previsione con SVM
                #Training Accuracy
                total_observations_training_misclassified = len([i for i in observations_training if value(z[i]) > 1])
                training_error_rate = total_observations_training_misclassified / len(observations_training)
                training_accuracy = 1 - training_error_rate
                
                #Testing Accuracy
                total_observations_testing_misclassified = np.sum(np.array(y_pred_test) != y_test)
                testing_error_rate = total_observations_testing_misclassified / len(observations_testing)
                testing_accuracy = 1 - testing_error_rate
                #accuracy.append(testing_accuracy)
                
                #Confusion Matrix
                cm = confusion_matrix(y_test, y_pred_test)
                TN = cm[0,0]
                FP = cm[0,1]
                FN = cm[1,0]
                TP = cm[1,1]
                #acc = (TP+TN)/(TP+TN+FN+FP)
                sens = TP/(TP+FN)
                spec = TN/(TN+FP)
                prec = TP/(TP+FP)


                #Salvo i risultati in un dizionario
                results.append({
                    'Try': try_number,
                    'Total Training Observations': len(observations_training),
                    'Total Testing Observations': len(observations_testing),
                    'Training Error Rate': training_error_rate,
                    'Training Accuracy': training_accuracy,
                    'Testing Error Rate': testing_error_rate,
                    'Testing Accuracy': testing_accuracy,
                    'Sensitivity': sens,
                    'Specificity': spec,
                    'Precision': prec
                })
            tok = time.time()
            
            #Converto i risultati in un dataframe
            results_df = pd.DataFrame(results)

            #Calcolo le medie per ogni metrica calcolata su 100 diverse combinazioni di dati
            averages = {
                'Mean Training Error Rate': results_df['Training Error Rate'].mean(),
                'Mean Training Accuracy': results_df['Training Accuracy'].mean(),
                'Mean Testing Error Rate': results_df['Testing Error Rate'].mean(),
                'Mean Testing Accuracy': results_df['Testing Accuracy'].mean(),
                'Mean Sensitivity': results_df['Sensitivity'].mean(),
                'Mean Specificity': results_df['Specificity'].mean(),
                'Mean Precision': results_df['Precision'].mean(),
                'Mean Standard Deviation Accuracy': results_df['Testing Accuracy'].std(),
                'Time': tok-tik
            }

            #Converto le medie in un dataframe
            averages_df = pd.DataFrame([averages])

            #Salvo in un file .xlsx
            file_name = f"svm_results_train{int(training_percentage*100)}_{dataset}_{C}.xlsx"

            with pd.ExcelWriter(file_name) as writer:
                #results_df.to_excel(writer, sheet_name="Results", index=False)
                averages_df.to_excel(writer, sheet_name="Averages", index=False)

            #Feedback di salvataggio completato con successo
            print(f"Results saved to {file_name}")
