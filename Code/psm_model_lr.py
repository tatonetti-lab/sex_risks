import numpy as np
import pandas as pd 
import feather 

from scipy.sparse import hstack, coo_matrix, save_npz, load_npz

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

import pymysql
import pymysql.cursors
from database import Database

from utils import Utils

db = Database('Mimir from Munnin')
u = Utils()
np.random.seed(u.RANDOM_STATE)

# put together features into X, y sparse matrices

df_patients = u.load_df('df_patients')
df_patients = df_patients.sort_values(by='PID')

drugs = u.load_np('drugs')

drug_features = []
for i, drugID in enumerate(drugs): 
    f = u.load_feature(drugID)
    drug_features.append(f)
    
age_feature = coo_matrix(df_patients.get('Age').values.reshape(u.NUM_PATIENTS, 1))

num_drugs_feature = coo_matrix(hstack(drug_features).sum(1))

features = []
features.append(age_feature)
features.append(num_drugs_feature)
features.extend(drug_features)

X = hstack(features)
y = u.load_feature('label').toarray().reshape(-1)

# random forest 

model = Pipeline([
                    ('feature_selection', SelectPercentile()),
                    ('classifier', LogisticRegression())
                ])

param_grid = { 
    'feature_selection__percentile' : [1, 5, 10],
    'feature_selection__score_func' : [chi2],
    'classifier__penalty': ['l1', 'l2']}

CV = GridSearchCV(model, param_grid, n_jobs=1)

CV.fit(X, y)  
print(CV.best_params_)    
print(CV.best_score_)

joblib.dump(CV, u.DATA_PATH+'PSM_models/LR/Grid_Search.pkl')
joblib.dump(CV.best_estimator_, u.DATA_PATH+'PSM_models/LR/Best.pkl')
