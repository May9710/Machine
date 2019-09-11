#!/usr/bin/env python
# coding: utf-8

#get_ipython().system('pip install --upgrade pip')
#get_ipython().system('pip install --upgrade tensorflow')
#get_ipython().system('pip install --upgrade keras')

# Importar librerías
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Fijar una semilla
random.seed(42)

# Cargar base de datos
dataset = pd.read_csv('TSLA.csv')
#dataset = pd.read_csv('AMZN.csv')
#dataset = pd.read_csv('FB.csv')
#dataset = pd.read_csv('NVDA.csv')
dataset = dataset.dropna()
dataset = dataset[['Open', 'High', 'Low', 'Close']]

# Crear columnas con promedios mensuales, entre otros
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['3day MA'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10day MA'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['30day MA'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_dev']= dataset['Close'].rolling(5).std()

# Valor de salida como aumento de precio (dummie) 1 cuando el precio de cierre de mañana
# es mayor que el precio de cierre de hoy.
dataset['Price_Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)
dataset = dataset.dropna()

# División del conjunto de datos en el conjunto de entrenamiento y el conjunto de prueba.
X = dataset.iloc[:, 4:-1]
y = dataset.iloc[:, -1]
split = int(len(dataset)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

# Feature Scaling - El escalado de características es un método utilizado para estandarizar
# el rango de variables independientes o características de los datos. "Normalizamos los datos"
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Creación Red neuronal
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Crear las capas (inicial, oculta y final)
classifier = Sequential()
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Usar gradiente en descenso estocastico "adam", realizar 100 épocas con 10 lotes (calcular error y reajustar los pesos)
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predecir el precio y guardarlo en una columna de la base de datos original
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred

# Crear una copia de la base de datos original donde se guardan los datos de la estrategia hallada
trade_dataset = dataset.dropna()
trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)
trade_dataset['Strategy Returns'] = 0.
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['Tomorrows Returns'], - trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

#Gráfica comparativa
plt.figure(figsize=(10,5))
plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
plt.legend()
plt.show()
