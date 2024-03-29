#!/usr/bin/env python
# coding: utf-8
get_ipython().system('pip install --upgrade pip')
get_ipython().system('pip install --upgrade tensorflow')
get_ipython().system('pip install --upgrade keras')

# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar base de datos
#dataset = pd.read_csv('AMZN.csv')
#dataset = pd.read_csv('FB.csv')
#dataset = pd.read_csv('NVDA.csv')
dataset = pd.read_csv('TSLA.csv')
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
# units → Número de nodos en la capa.
# Kernel_initializer → pesos con distribución uniforme.
# activation → rectified Linear Unit function.
# input_dim → número de entradas a la capa (número de columnas X) 
# optimizer → extensión del gradiente en descenso estocástico (reducción función de costo)
# loss → Pérdida a optimizar (error cuadrático medio)
# metrics → Precisión (número de predicciones correctas / número de predicciones totales)
# batch_size → Cantidad de datos que el modelo usa para calcular el error con los que se modifican los pesos.
# epochs → Número de épocas (veces que se realiza el entrenamiento del modelo en el conjunto de datos)
classifier = Sequential()
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = X.shape[1]))
classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predecir el precio y guardarlo en una columna de la base de datos original
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
trade_dataset = dataset.dropna()

# Crear una copia de la base de datos original donde se guardan los datos de la estrategia hallada
# Logaritmo del precio de cierre de hoy dividido por el precio de cierre de ayer.
trade_dataset['Tomorrows Returns'] = 0.
trade_dataset['Tomorrows Returns'] = np.log(trade_dataset['Close']/trade_dataset['Close'].shift(1))
trade_dataset['Tomorrows Returns'] = trade_dataset['Tomorrows Returns'].shift(-1)
trade_dataset['Strategy Returns'] = 0.
trade_dataset['Strategy Returns'] = np.where(trade_dataset['y_pred'] == True, trade_dataset['Tomorrows Returns'], - trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Market Returns'] = np.cumsum(trade_dataset['Tomorrows Returns'])
trade_dataset['Cumulative Strategy Returns'] = np.cumsum(trade_dataset['Strategy Returns'])

#Gráfica comparativa
#plt.figure(figsize=(10,5))
#plt.title('AMAZON')
#plt.plot(trade_dataset['Cumulative Market Returns'], color='r', label='Market Returns')
#plt.plot(trade_dataset['Cumulative Strategy Returns'], color='g', label='Strategy Returns')
#plt.legend()
#plt.savefig('amazon.jpg')
#plt.show()

#plt.figure(figsize=(10,5))
#plt.title('FACEBOOK')
#plt.plot(trade_dataset['Cumulative Market Returns'], color='magenta', label='Market Returns')
#plt.plot(trade_dataset['Cumulative Strategy Returns'], color='darkviolet', label='Strategy Returns')
#plt.legend()
#plt.savefig('facebook.jpg')
#plt.show()

#plt.figure(figsize=(10,5))
#plt.title('NVIDIA')
#plt.plot(trade_dataset['Cumulative Market Returns'], color='gold', label='Market Returns')
#plt.plot(trade_dataset['Cumulative Strategy Returns'], color='saddlebrown', label='Strategy Returns')
#plt.legend()
#plt.savefig('nvidia.jpg')
#plt.show()

plt.figure(figsize=(10,5))
plt.title('TESLA')
plt.plot(trade_dataset['Cumulative Market Returns'], color='b', label='Market Returns')
plt.plot(trade_dataset['Cumulative Strategy Returns'], color='c', label='Strategy Returns')
plt.legend()
plt.savefig('tesla.jpg')
plt.show()

inicial = 10000
df = trade_dataset[['y_pred','Strategy Returns']]
df = df.drop(df[df['y_pred']==False].index)
def f(x):
    return inicial * x
df['Final'] = df['Strategy Returns'].apply(f)
df['acumulado'] = np.cumsum(df['Final'])
df = df.dropna()
value_final = df['acumulado'].iloc[-1]
value_final

#El presente codigo no es un metodo efectivo de inversion ya que genera perdidas a la hora de invertir, sin embargo presenta
#oportunidades de entrar al mercado al contar con informacion relevante para predecir el movimiento de la tendencia, como se ve 
#en cada uno de los resultados obtenidos, los retornos del mercado y los retornos de la estrategia del codigo coinciden en 
#varios intervalos de tiempo la tendencia de los mismo. 
#Al momento de realizar una inversion, no es efectivo pues no genera ganancias significativas. 
