import sys

import numpy as np
import pandas as pd

from scipy.sparse import hstack, coo_matrix, save_npz, load_npz

import pymysql
import pymysql.cursors
from database import Database

from utils import Utils
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import itertools

db = Database('awaredx')
u = Utils()

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

MYSQL_DB_ADX = config['DATABASE']['mysql_db_awaredx']
MYSQL_DB_FDA = config['DATABASE']['mysql_db_openfda']

def make_features(drugID):

    # i = sys.argv[1]
    # i = int(i)

    q = f"""
        SELECT 
            COUNT(CASE WHEN atc_5_id = {drugID} THEN 1 END) AS drug
        FROM """+MYSQL_DB_ADX+""".atc_5_patient_psm psm
        JOIN """+MYSQL_DB_FDA+""".patient p
            ON psm.PID = p.safetyreportid
        WHERE (p.patient_sex='Female' OR p.patient_sex='Male')
            AND p.patient_custom_master_age BETWEEN 18 AND 85
        GROUP BY psm.PID
        ORDER BY psm.PID
    """

    # q = 'select count(case when atc_5_id = ' + str(
    #     drugID
    # ) + ' then 1 end) as drug from atc_5_patient_psm group by PID order by PID'

    # drug_feature = np.array(db.get_list(q)).reshape(u.NUM_PATIENTS, 1)
    drug_feature = np.array(db.get_list(q)).reshape(-1, 1)

    u.save_feature(drug_feature, drugID)


def make_features_mp():

    drugs = u.load_np('drugs')

    process_map(make_features, drugs, max_workers=10)


if __name__ == '__main__':
    make_features_mp()