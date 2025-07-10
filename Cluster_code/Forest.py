import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import QuantileTransformer
import logging
from joblib import dump, load
from time import time
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action="ignore", category=DataConversionWarning)

t0 = time()

# -----------
# load
X_train = pd.read_csv("xtrain.csv", index_col=0)
y_train = pd.read_csv("ytrain.csv", index_col=0)
X_test = pd.read_csv("xtest.csv", index_col=0)
y_test = pd.read_csv("ytest.csv", index_col=0)
groups = X_train['groups']
X_train = X_train.drop(columns='groups')

# -----------
# Pipeline
pipe = Pipeline([
    ('transformer', QuantileTransformer()),
    ('regressor', RandomForestRegressor(random_state=42))
])

# CV object
CV = GroupKFold(n_splits=10)

# gridsearch parameters
regressor_par = {
    'regressor__max_depth': [10,15,20],
    'regressor__max_features': [12,15,18],
    'regressor__max_leaf_nodes': [1000, 2000, 3000],
    'regressor__min_samples_leaf': np.arange(1, 3),
    'regressor__min_samples_split': [2,3,4],
    'regressor__n_estimators': [300, 400, 500]
}

param_grid = [{**regressor_par}]

# Instantiate the grid search model
grid_search = GridSearchCV(pipe, param_grid=param_grid,
                           cv=CV, n_jobs=-1, verbose=1, refit=True, error_score='raise')

# ------------------------------------
# train
model = grid_search.fit(X_train, y_train, groups=groups)
best_model = model.best_estimator_


# save
dump(model, 'forest.joblib')


# -----------------------------------
# permutation feature importance
feature_names = list(X_train.columns)
result = permutation_importance(
    best_model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=-1
)

# sort importances
sort_idx = result.importances_mean.argsort()
importances_mean = result.importances_mean[sort_idx]
importances_std = result.importances_std[sort_idx]
feature_names = np.array(feature_names)[sort_idx]
forest_importances = pd.Series(importances_mean, index=feature_names)
top_feat = feature_names[::-1][:20]

# plot permutation feature importances for top feats
fig, ax = plt.subplots(figsize=(16, 12))
forest_importances[top_feat[::-1]].plot.barh(ax=ax)
ax.set_title("Permutation feature importances", fontsize=20)
ax.set_xlabel("Mean r2 decrease", fontsize=20)
for item in (ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(20)
fig.tight_layout()
plt.savefig('forest_FI.png')

# save feature importances
forest_importances.to_csv('forest_FI.csv')


# ----------------------------------------
# finished
print('Finished in {}'.format(time()-t0))
