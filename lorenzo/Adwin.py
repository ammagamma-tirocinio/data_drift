from bike_demand_analysis import raw_data
from river.datasets import AirlinePassengers
from tools import show_plot, adwin_detection
from iris_dataset import virginica, setosa, versicolor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


## Bike Sharing Demand Kaggle's Dataset

data_stream = raw_data.loc[:,'cnt']
predicted_indexes = adwin_detection(data_stream)
show_plot(data_stream, predicted_indexes, syntetic_dataset = False)
plt.title('Bike-Demand Data drift with ADWIN')
plt.show()

quit()
# syntetic dataset - gradual drift

data_stream = np.concatenate((np.random.uniform(1,10,100),np.random.uniform(10,30,50), np.random.uniform(30,50,50), np.random.uniform(10,15,20)))
real_indexes = [100,150,200]
predicted_indexes = adwin_detection(data_stream)
#show_plot(data_stream,predicted_indexes,real_indexes)
plt.title('Gradual Data drift with ADWIN')


# syntetic iris dataset
data_stream = np.concatenate((virginica.iloc[:50,0],setosa.iloc[:50,0],versicolor.iloc[:10,0],virginica.iloc[20:25,0]))
predicted_indexes = adwin_detection(data_stream)
real_indexes = [50,100,110,115]
#show_plot(data_stream,predicted_indexes,real_indexes)
plt.title('Iris syntetic dataset Data drift with ADWIN')


# syntetic dataset - abrut drift

data_stream = np.concatenate((np.random.uniform(10,30,100),np.random.uniform(150,200,50), np.random.uniform(30,50,50), np.random.uniform(10,15,20)))
real_indexes = [100,150]
predicted_indexes = adwin_detection(data_stream)
#show_plot(data_stream, predicted_indexes, real_indexes)
#plt.title('Abrut Data drift with ADWIN')

## Airline Demand Kaggle's Dataset
data_stream = pd.read_csv(AirlinePassengers().path).iloc[:,-1]
predicted_indexes = adwin_detection(data_stream)
#show_plot(data_stream, predicted_indexes)
#plt.title('Airplane Data drift with ADWIN')

plt.show()
