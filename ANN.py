#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:31:12 2019

@author: valeriabz
"""



#La biblioteca de Keras es una Biblioteca increíble para construir modelos de aprendizaje profundo Como redes neuronales profundas en muy pocas líneas de código.
# Y además, son muy eficientes. 

# Classification template

# Importar las librerias 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar las bases de datos
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Cambiar la codificación de los datos. 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) #Columna uno - Nombre de los paises
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) #Columna dos - Sexo del cliente
'''
 The Dummy Variable trap is a scenario in which the independent variables are multicollinear - 
 a scenario in which two or more variables are highly correlated; in simple terms one variable 
 can be predicted from the others.
'''
onehotencoder = OneHotEncoder(categorical_features = [1]) #QUitar una variable dummy de la primera columna para evitar la trampa del Dummy
X = onehotencoder.fit_transform(X).toarray() 
X = X[:,1:] #Quita la primera columna (Columna 0)

# División del conjunto de datos en el conjunto de entrenamiento y el conjunto de prueba.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0) #El 20% de la base de datos lo pone como muestra

# Feature Scaling - El escalado de características es un método utilizado 
#para estandarizar el rango de variables independientes o características de los datos. "Normalizamos los datos"
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Part 2 - Hagamos la ANN!
import keras 
from keras.models import Sequential  #Para inicial la ANN
from keras.layers import Dense #Crear las capas 

#Iniciar la ANN 
classifier = Sequential()
#crear la capa de entrada y la primera capa oculta 
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation="relu", input_dim = 11 ))
#Pongo 6 porque son el numero de nodos de la capa de entrada 11 divido en la mitad, iniciamos la función uniforme porque estamos buscando valores 
#cercanos a 0 y la función de activación va a ser la rectificadora (relu)

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Crear la segunda capa oculta (Aqui usamos los mismos parametros anteriores pero ya no contamos con input_dim)
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

#Crear la capa de salida (Queremos solo 1 nodo y una estimación de probabilidad, por lo que usamos la función sigmoide)
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compilar la ANN (Usando el gradiente en descenso estocastico "adam")
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Dividir la ANN en el conjunto de entrenamiento (Hacemos 100 actualizaciones en los pesos para mejorar el aprendizaje)
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Hacer las predicciones y evaluar el modelo 

# Predecir el conjunto de resultados (Lo que queriamos!)
y_pred = classifier.predict(X_test) 
y_pred = (y_pred > 0.5) 

# Hacer la matriz de confusión (Para determinar porcentajes)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

'''
Pasos para crear la ANN con el gradiente en descenso estocasticos 
1. Iniciar los pesos de las capas aleatoriamente con numeros cercanos a 0

2. ingresar la primera observación del conjunto de datos en la capa de entrada, cada variable en un nodo de entrada. 
En este caso, hay 11 variables independientes, por lo tanto, hay 11 nodos. 

3. Propagación hacia adelante: de izquierda a derecha, las neuronas se activan de manera que el impacto de la función activación 
en cada neurona está limitado por los pesos. Propague las activaciones hasta obtener el resultado  y.
En este caso, usaremos la función de activación: Rectificador, para las capas ocultas y se usará la función sigmoide para la capa 
de salida 

4. Comparar el resultado obtenido con el resultado real para medir el error generado.
 
5. Propagación hacia atrás: de derecha a izquierda, el error generado es propagado hacia atrás. Actualizar los pesos de acuerdo a
 cuánto son responsables del error cada peso. La tasa de aprendizaje decide por cuánto actualizamos los pesos.
 
 6. Repetir los pasos 1 a 5 y actualizar los pesos después de cada observación (Aprendizaje reforzado) o repetir los pasos 1 a 5 
 pero actualizando los pesos después de cada lote de observación (Aprendizaje por lotes)

7. Hacemos muchas comprobaciones (epoch)
'''
