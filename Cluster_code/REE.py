import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.inspection import permutation_importance
from joblib import dump, load
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore")
import json

# -----------
# load data
X_train = pd.read_csv("xtrain.csv", index_col=0)
y_train = pd.read_csv("ytrain.csv", index_col=0)
X_test = pd.read_csv("xtest.csv", index_col=0)
y_test = pd.read_csv("ytest.csv", index_col=0)
groups = X_train["groups"]
X_train = X_train.drop(columns="groups")

# model
model = load('forest.joblib')
best_model = model.best_estimator_

# loop and remove 
top_feat_list = []
for i in range(15):
    # fit
    best_model.fit(X_train,y_train)

    # permutation feature importance
    feature_names = list(X_train.columns)
    result = permutation_importance(
        best_model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
    )
    sort_idx = result.importances_mean.argsort()[::-1]
    feature_names = np.array(feature_names)[sort_idx]
    
    # top feature
    top_feat = feature_names[0]
    top_feat_list.append(top_feat)
    
    # remove effect of top feature
    pipe = Pipeline([
    ('transformer', QuantileTransformer()),
    ('regressor', RandomForestRegressor(random_state=42))
    ])    
    X = X_train[top_feat].values.reshape(-1, 1)
    pipe.fit(X,y_train)
    X_train = X_train.drop(columns=top_feat)
    y_train = y_train - pipe.predict(X).reshape(-1,1)
    
# retrain on the top features selected 
# -----------------------
# reload
X_train = pd.read_csv("xtrain.csv", index_col=0)
y_train = pd.read_csv("ytrain.csv", index_col=0)
X_test = pd.read_csv("xtest.csv", index_col=0)
y_test = pd.read_csv("ytest.csv", index_col=0)
groups = X_train["groups"]
X_train = X_train.drop(columns="groups")

# select features
X_train = X_train[top_feat_list]
X_test = X_test[top_feat_list]

# fit
best_model.fit(X_train,y_train)

# save model
dump(best_model, 'forest_ree.joblib')

# Print output
with open("REE_list.txt", "w") as fp:
    json.dump(top_feat_list, fp)
print('Validation score:\ntraining set: r2 = {:3.2f}\ntest set: r2 = {:3.2f}'.format(best_model.score(X_train, y_train),
                                                                           best_model.score(X_test, y_test)))
# permutation feature importance
feature_names = list(X_train.columns)
result = permutation_importance(
    best_model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
)
sort_idx = result.importances_mean.argsort()
importances_mean = result.importances_mean[sort_idx]
importances_std = result.importances_std[sort_idx]
feature_names = np.array(feature_names)[sort_idx]
forest_importances = pd.Series(importances_mean, index=feature_names)

#plot permutation feature importances for top feats
fig, ax = plt.subplots(figsize=(16, 12))
forest_importances.plot.barh(ax=ax)
ax.set_title("Permutation feature importances", fontsize=20)
ax.set_xlabel("Mean r2 decrease", fontsize=20)
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)
fig.tight_layout()
plt.savefig('forest_REE_FI.png')

# save feature importances
forest_importances.to_csv('REE_FI.csv')
