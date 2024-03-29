from utils import Utils
from database import Database
import pandas as pd

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

u = Utils()
db = Database('awaredx')

MYSQL_DB_FDA = config['DATABASE']['mysql_db_openfda']


def fetch_patients(date_filter=None, age_filter=85):
    '''
    date_filter: leave empty if not wanted, otherwise list of two strings with 2 dates in 'yyy-mm-dd' format
    '''

    # get patients that have sex
    if date_filter:
        
        q_patients_w_sex = """
        SELECT pa.safetyreportid as PID
        , LEFT(pa.patient_sex, 1) as Sex
        , pa.patient_custom_master_age as Age
        FROM """+MYSQL_DB_FDA+""".patient  pa
        LEFT JOIN effect_openfda_2023q4.report re 
        ON pa.safetyreportid = re.safetyreportid
        WHERE (pa.patient_sex='Female' OR pa.patient_sex='Male')
        AND pa.patient_custom_master_age BETWEEN 18 AND @age_filter
        AND re.receive_date BETWEEN """+date_filter[0]+ """ AND """ +date_filter[1]
    else:
        q_patients_w_sex = """
            SELECT safetyreportid as PID
            , LEFT(patient_sex, 1) as Sex
            , patient_custom_master_age as Age
            FROM """+MYSQL_DB_FDA+""".patient 
            WHERE (patient_sex='Female' OR patient_sex='Male')
            AND patient_custom_master_age BETWEEN 18 AND """ + str(age_filter)
    
    print(q_patients_w_sex)

    res_patients_w_sex = db.run_query(q_patients_w_sex)

    # make dataframe
    df_patients = pd.DataFrame(res_patients_w_sex,
                               columns=['PID', 'Sex', 'Age'])

    # replace 'Male' 'Female' with M, F
    # df_patients = df_patients.replace('Female', 'F').replace('Male', 'M')

    # drop missing ages
    # df_patients = df_patients.patients.replace('', np.nan)
    # df_patients = df_patients.dropna(subset=['Age'])
    # df_patients = df_patients.astype({'Age': 'int'})

    # remove age below 18 and above 85
    # df_patients = df_patients.query('Age>=18 and Age<=85')

    # Save patients
    u.save_df(df_patients, 'df_patients')

    return df_patients


if __name__ == '__main__':
    fetch_patients()