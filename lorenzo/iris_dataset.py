from sklearn import datasets
import pandas as pd
import numpy as np

iris = datasets.load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

setosa = iris_df[iris_df.target == 0.0]
versicolor = iris_df[iris_df.target == 1.0]
virginica = iris_df[iris_df.target == 2.0]


