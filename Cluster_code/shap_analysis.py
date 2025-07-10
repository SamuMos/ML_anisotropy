import warnings
import json
warnings.simplefilter(action="ignore", category=FutureWarning)
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)

import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd
import sklearn

from joblib import dump, load
import shap

# load data
X_train = pd.read_csv("xtrain.csv", index_col=0)
X_train = X_train.drop(columns="groups")

# select features
with open("REE_list.txt", "r") as fp:
    ree_list = json.load(fp)
X_train = X_train[ree_list]

# scale
X_train = pd.DataFrame(
    sklearn.preprocessing.QuantileTransformer().fit_transform(X_train),
    index=X_train.index,
    columns=X_train.columns,
)

# load model
model = load("forest_ree.joblib").set_params(regressor__n_jobs=-1)

# shap values
X_sample = shap.maskers.Independent(X_train, max_samples=100)
explainer = shap.Explainer(model[-1], X_sample, seed=42)
shap_values = explainer(X_train)

# save
dump(shap_values, 'shap_values.joblib')

