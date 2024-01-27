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

# def make_features():
#     drugs = u.load_np('drugs')

#     q = """
#         SELECT
#             atc_5_id, PID, COUNT(*) drug
#         FROM atc_5_patient_psm p
#         WHERE atc_5_id IN (21604154, 21603927, 21604711, 21600449, 21604348)
#         GROUP BY p.atc_5_id, PID
#         ORDER BY PID
#     """

#     matched_atc_pids = pd.DataFrame(db.run_query(q),
#                                     columns=['atc_5_id', 'PID', 'drug'])

#     unique_pids = sorted(matched_atc_pids['PID'].unique())
#     print(len(drugs), len(unique_pids))

#     for drugID in tqdm(drugs):
#         # atc_pids_combos = list(itertools.product([drugID], unique_pids))

#         # atc_pids_combos_df = pd.DataFrame(atc_pids_combos,
#         #                                   columns=['atc_5_id', 'PID'])

#         atc_pids_combos_df = pd.Series(unique_pids, name='PID').to_frame()
#         atc_pids_combos_df['atc_5_id'] = drugID

#         all_atc_pids = atc_pids_combos_df.merge(matched_atc_pids,
#                                                 on=['atc_5_id', 'PID'],
#                                                 how='left').fillna(0)

#         drug_feature = np.array(all_atc_pids['drug']).reshape(
#             len(unique_pids), 1)

#         u.save_feature(drug_feature, drugID)

#     # print("producting itertools")
#     # atc_pids_combos = list(itertools.product(drugs, unique_pids))
#     # print("done producting itertools")
#     # print(len(atc_pids_combos))
#     # atc_pids_combos_df = pd.DataFrame(atc_pids_combos,
#     #                                   columns=['atc_5_id', 'PID'])

#     # print("merging")
#     # all_atc_pids = atc_pids_combos_df.merge(matched_atc_pids,
#     #                                         on=['atc_5_id', 'PID'],
#     #                                         how='left')
#     # all_atc_pids = all_atc_pids.fillna(0)

#     # all_atc_pids = all_atc_pids.sort_values(by=['atc_5_id', 'PID'])

#     # print("saving features")
#     # for drugID in tqdm(drugs):
#     #     drug_feature = np.array(
#     #         all_atc_pids[all_atc_pids['atc_5_id'] == drugID]['drug'])

#     #     u.save_feature(drug_feature, drugID)


def make_features(drugID):

    # i = sys.argv[1]
    # i = int(i)

    q = f"""
        SELECT 
            COUNT(CASE WHEN atc_5_id = {drugID} THEN 1 END) AS drug
        FROM atc_5_patient_psm psm
        JOIN effect_openfda_19q2.patient p
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