import numpy as np
import joblib

from database import Database
from utils import Utils

db = Database('awaredx')
u = Utils()
np.random.seed(u.RANDOM_STATE)


def calculate_propensity_scores():
    df_patients = u.load_df('df_patients')
    df_patients = df_patients.sort_values(by='PID')

    model = joblib.load(u.DATA_PATH + 'PSM_models/RF2/Best.pkl')
    clf = model['classifier']

    p_scores = clf.oob_decision_function_[:,
                                          1]  # 1 is idx where clf.classes_ == 1
    df_patients['Propensity'] = p_scores

    u.save_df(df_patients, 'df_propensity')


if __name__ == '__main__':
    calculate_propensity_scores()