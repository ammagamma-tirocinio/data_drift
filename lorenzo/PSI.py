from bike_demand_analysis import raw_data
from tools import show_plot, psi_detection
from iris_dataset import virginica, setosa, versicolor
from river.datasets import AirlinePassengers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## syntetic dataset - gradual drift

#Dato il giorno t confronto la distr

data_stream = np.concatenate((np.random.uniform(1,10,100),np.random.uniform(10,30,50), np.random.uniform(30,50,50), np.random.uniform(10,15,20)))
real_indexes = [100,150,200]
predicted_index_high, predicted_index_mid = psi_detection(data_stream, w0 = 20, w1 = 20)
show_plot(data_stream,predicted_index_high,real_indexes)
plt.title('Gradual Data drift with  PSI > 0.3')
# show_plot(data_stream,predicted_index_mid,real_indexes)
# plt.title('Gradual Data drift with 0.2 > PSI < 0.3')
plt.show()

# syntetic iris dataset
data_stream = np.concatenate((virginica.iloc[:50,0],setosa.iloc[:50,0],versicolor.iloc[:10,0],virginica.iloc[20:25,0]))
predicted_index_high, predicted_index_mid = psi_detection(data_stream, w0 = 20, w1 =20)
show_plot(data_stream,predicted_index_high,real_indexes)
plt.title('Iris syntetic dataset Data drift with  PSI > 0.3')
plt.show()

## syntetic dataset - abrut drift

data_stream = np.concatenate((np.random.uniform(10,30,100),np.random.uniform(150,200,50), np.random.uniform(30,50,50), np.random.uniform(10,15,20)))
real_indexes = [100,150]
predicted_index_high, predicted_index_mid = psi_detection(data_stream, w0 = 20, w1 = 20)
show_plot(data_stream,predicted_index_high,real_indexes)
plt.title('Gradual Data drift with  PSI > 0.3')
plt.show()

## Bike Sharing Demand Kaggle's Dataset

data_stream = raw_data.loc[:,'cnt'].rolling(window =20).mean()
predicted_index_high, predicted_index_mid = psi_detection(data_stream, w0 = 20, w1 = 20)
show_plot(data_stream, predicted_index_high, syntetic_dataset = False)
plt.title('Bike Demand Data drift with  PSI > 0.3')
plt.show()

## Airline Demand Kaggle's Dataset
data_stream = pd.read_csv(AirlinePassengers().path).iloc[:,-1]
predicted_index_high, predicted_index_mid  = psi_detection(data_stream,w0 = 20, w1 = 20)
show_plot(data_stream, predicted_index_high)
plt.title('Airplane Data drift with  PSI > 0.3')

plt.show()
