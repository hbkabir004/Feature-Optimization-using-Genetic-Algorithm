import numpy as np
import pandas as pd
import os, sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import random
import math
from pandas import Series
from pyspark.sql import Row

# Read Dataset
df = pd.read_csv("data.csv")
d = {'M': 1, 'B': 0}
df['diagnosis'] = df['diagnosis'].map(d)
df=df.T.drop_duplicates().T

# Missing Value Imputation
#df.ca.loc[df.ca=='?']     = '0' 
#df.thal.loc[df.thal=='?'] = '3'

# Remove Outlier
#df = df[df.chol<500][df.oldpeak<5]

# Remove Unnecessary Column
df.drop(columns = 'id', axis = 1, inplace = True)

# One-Hot Encoding
df = pd.get_dummies(df, columns = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'], drop_first = True)

# Get features and target variables
target ='diagnosis'
target, feature_list = 'diagnosis', [i for i in df.columns if i not in target]

X = df.loc[:, df.columns != target]
y = df.loc[:, target]
# Print list of features and target variable names
print('Feature List\n',feature_list, '\n\nTarget = ',target)


def init_population(n,c):
    return np.array([[math.ceil(e) for e in pop] for pop in (np.random.rand(n,c)-0.5)]), np.zeros((2,c))-1

def single_point_crossover(population):
    r,c, n = population.shape[0], population.shape[1], np.random.randint(1,population.shape[1])         
    for i in range(0,r,2):                
        population[i], population[i+1] = np.append(population[i][0:n],population[i+1][n:c]),np.append(population[i+1][0:n],population[i][n:c])        
    return population

def flip_mutation(population):
    return population.max() - population

def random_selection(population):
    r = population.shape[0]
    new_population = population.copy()    
    for i in range(r):        
        new_population[i] = population[np.random.randint(0,r)]
    return new_population


def get_fitness(data, feature_list, target, population):    
    fitness = []
    for i in range(population.shape[0]):        
        columns = [feature_list[j] for j in range(population.shape[1]) if population[i,j]==1]                    
        fitness.append(predictive_model(data[columns], data[target]))                
    return fitness

def predictive_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=7)
    rfc = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
    rfc.fit(X_train,y_train)
    return accuracy_score(y_test, rfc.predict(X_test))

    predictions = rfc.predict(X_test)
    rfc_pred = rfc.predict(X_test)
    
    
    

    #rfc_score = accuracy_score(y_test,rfc_pred)
    #precission_rfc = precision_score(y_test,rfc_pred)
    #recall_rfc = recall_score(y_test,rfc_pred)
    #f1_rfc = f1_score(y_test,rfc_pred)
    
    #return ("Accuracy_Score by cross_val_orediction :", Accuracy_Score)
    #return ("Precission : ",Precision_Score)
    #return ("Recall : ",Recall_Score)
    #return ("F1 : ",f1_Score)



def ga(data, feature_list, target, n, max_iter):

    c = len(feature_list) 
    
    population, memory = init_population(n,c)
     #def replace_duplicate(population, memory) :
   # population, memory =replace_duplicate(population, memory)    
    
    fitness= get_fitness(data, feature_list, target, population)    
    
    optimal_value= max(fitness)
    optimal_solution = population[np.where(fitness==optimal_value)][0]    
    
    for i in range(max_iter):                
        population = random_selection(population)
        population = single_point_crossover(population)                        
        if np.random.rand() < 0.3:
            population = flip_mutation(population)   
        
      #  population, memory = replace_duplicate(population, memory)
                
        fitness = get_fitness(data, feature_list, target, population)
                
        if max(fitness) > optimal_value:
            optimal_value    = max(fitness)
            optimal_solution = population[np.where(fitness==optimal_value)][0]                               
        
            return optimal_solution, optimal_value


# Execute Genetic Algorithm to obtain Important Feature
feature_set, acc_score= ga(df, feature_list, target, 20, 1000)

# Filter Selected Features
feature_set = [feature_list[i] for i in range(len(feature_list)) if feature_set[i]==1]

# Print List of Features
print('Optimal Feature Set\n',feature_set,'\nOptimal Accuracy =', round(acc_score*100), '%')

#print('Average Accuracy saved', Accuracy_Score, '\n Average Precision', Precision_Score, '\n Average Recall',Recall_Score,'\n Average F1-Score',  F1_Score)


rfc = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
scores = cross_val_score(estimator=rfc, X=X, y=y, cv=10, scoring='accuracy')
predicted_label = cross_val_predict(estimator=rfc, X=X, y=y, cv=10)
score = round(scores.mean() * 100, 4)
#print(score)
Accuracy_Score = accuracy_score(y, predicted_label)
Precision_Score = precision_score(y, predicted_label, average="macro")
Recall_Score = recall_score(y, predicted_label, average="macro")
F1_Score = f1_score(y, predicted_label, average="macro")
print('Average Accuracy saved', Accuracy_Score, '\n Average Precision', Precision_Score, '\n Average Recall',Recall_Score,'\n Average F1-Score',  F1_Score)


import matplotlib.pyplot as plt
model = RandomForestClassifier()
model.fit(X,y)
print(model.feature_importances_) 
#use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
      
        