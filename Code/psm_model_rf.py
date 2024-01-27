import numpy as np
import pandas as pd
import pyarrow.feather as feather

from scipy.sparse import hstack, coo_matrix, save_npz, load_npz

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, f_classif, SelectPercentile
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

import pymysql
import pymysql.cursors
from database import Database

from utils import Utils

#db = Database('Mimir from Munnin')
u = Utils()
np.random.seed(u.RANDOM_STATE)


def run_rf_model():

    # put together features into X, y sparse matrices

    df_patients = u.load_df('df_patients')
    df_patients = df_patients.sort_values(by='PID')

    drugs = u.load_np('drugs')

    drug_features = []
    for i, drugID in enumerate(drugs):
        f = u.load_feature(drugID)
        drug_features.append(f.astype(float))

    # age_feature = coo_matrix(
    #     df_patients.get('Age').values.reshape(u.NUM_PATIENTS, 1))

    age_feature = coo_matrix(df_patients.get('Age').values.reshape(-1, 1))

    num_drugs_feature = coo_matrix(hstack(drug_features).sum(1))

    features = []
   

    features.append(age_feature.astype(float))
    features.append(num_drugs_feature.astype(float))
    features.extend(drug_features)
    

    X = hstack(features, dtype=float)
    # y = u.load_feature('label').toarray().reshape(-1)

    le = LabelEncoder()
    le.fit(['M', 'F'])
    y = le.transform(df_patients['Sex'].array)

    # random forest

    model = Pipeline([('feature_selection', SelectPercentile()),
                      ('classifier', RandomForestClassifier())])

    param_grid = {
        'feature_selection__percentile': [1, 5, 10],
        'feature_selection__score_func': [chi2],
        'classifier__oob_score': [True],
        'classifier__n_estimators': [50, 100, 150, 200],
        'classifier__max_depth': [5, 7, 9],
        'classifier__criterion': ['gini', 'entropy']
    }

    CV = GridSearchCV(model, param_grid, n_jobs=30)

    print('Fitting model')

    CV.fit(X, y)
    print(CV.best_params_)
    print(CV.best_score_)
    print(CV.best_estimator_['classifier'].oob_score_)

    joblib.dump(CV, u.DATA_PATH + 'PSM_models/RF2/Grid_Search.pkl')
    joblib.dump(CV.best_estimator_, u.DATA_PATH + 'PSM_models/RF2/Best.pkl')
    joblib.dump(CV.best_estimator_['classifier'].classes_,
                u.DATA_PATH + 'PSM_models/RF2/Best_Classes.pkl')
    joblib.dump(CV.best_estimator_['classifier'].oob_decision_function_,
                u.DATA_PATH + 'PSM_models/RF2/Best__Oob_Decision_Function.pkl')
    joblib.dump(CV.best_estimator_['classifier'].oob_score_,
                u.DATA_PATH + 'PSM_models/RF2/Best__Oob_Score.pkl')


if __name__ == '__main__':
    run_rf_model()