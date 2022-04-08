import shap
from bike_demand_analysis import d1, d2, train, val, model

def shap_plot(data,model):
    shap.initjs()
    data_summary = shap.kmeans(data, 10)
    explainer = shap.KernelExplainer(model.predict,data_summary)
    shap_values = explainer.shap_values(data)
    return shap.summary_plot(shap_values, features = data, feature_names = data.columns, plot_type= 'bar')

#Bike Demand - feature relevance: Train

x_train = train.iloc[:,:-1]
shap_plot(x_train, model)

#Bike Demand - feature relevance: Validation

x_val = val.iloc[:,:-1]
shap_plot(x_val, model)

##Bike Demand - feature relevance: Mesi primaverili/estivi

x_d2 = d2.iloc[:,:-1]
shap_plot(x_d2, model)


