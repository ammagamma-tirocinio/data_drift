import pandas as pd

from bike_demand_analysis import raw_data
from tools import outliers_detection, show_plot
from iris_dataset import virginica, setosa, versicolor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## syntetic iris dataset
data_stream = np.concatenate((virginica.iloc[:50,0],setosa.iloc[:50,0],versicolor.iloc[:10,0],virginica.iloc[20:25,0]))
predicted_indexes = outliers_detection(data_stream, 30)
real_indexes = [50,100,110,115]
show_plot(data_stream, predicted_indexes, real_indexes)
plt.title('Iris syntetic dataset Outlier detection')
plt.show()

## bike demand dataset

data_stream = raw_data.iloc[-200:,-1]
predicted_indexes = outliers_detection(data_stream, 30)
show_plot(data_stream, predicted_indexes, syntetic_dataset = False)
plt.title('Bike-Demand Outlier detection')
plt.show()

