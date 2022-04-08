from bike_demand_analysis import raw_data
from tools import show_plot, kswin_detection
from river.datasets import AirlinePassengers
from iris_dataset import virginica, setosa, versicolor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# syntetic dataset - gradual drift

data_stream = np.concatenate((np.random.uniform(1,10,100),np.random.uniform(10,30,50), np.random.uniform(30,50,50), np.random.uniform(10,15,20)))
real_indexes = [100,150,200]
predicted_indexes = kswin_detection(data_stream)
show_plot(data_stream,predicted_indexes,real_indexes)
plt.title('Gradual Data drift with KSWIN')
plt.show()

# syntetic iris dataset
data_stream = np.concatenate((virginica.iloc[:50,0],setosa.iloc[:50,0],versicolor.iloc[:10,0],virginica.iloc[20:25,0]))
predicted_indexes = kswin_detection(data_stream)
real_indexes = [50,100,110,115]
show_plot(data_stream,predicted_indexes,real_indexes)
plt.title('Iris syntetic dataset Data drift with KSWIN')
plt.show()

# syntetic dataset - abrut drift

data_stream = np.concatenate((np.random.uniform(10,30,100),np.random.uniform(150,200,50), np.random.uniform(30,50,50), np.random.uniform(10,15,20)))
real_indexes = [100,150]
predicted_indexes = kswin_detection(data_stream)
show_plot(data_stream, predicted_indexes, real_indexes)
plt.title('Abrut Data drift with KSWIN')
plt.show()

## Bike Sharing Demand Kaggle's Dataset

data_stream = raw_data.loc[:,'cnt']
predicted_indexes = kswin_detection(data_stream)
show_plot(data_stream, predicted_indexes, syntetic_dataset = False)
plt.title('Bike-Demand Data drift with KSWIN')

plt.show()

## Airline Demand Kaggle's Dataset
data_stream = pd.read_csv(AirlinePassengers().path).iloc[:,-1]

predicted_indexes = kswin_detection(data_stream)
show_plot(data_stream, predicted_indexes)
plt.title('Airplane Data drift with KSWIN')

plt.show()
