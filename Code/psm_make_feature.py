import sys

import numpy as np

from scipy.sparse import hstack, coo_matrix, save_npz, load_npz

import pymysql
import pymysql.cursors
from database import Database

from utils import Utils

i = sys.argv[1]
i = int(i)
db = Database('Mimir from Munnin')
u = Utils()

drugs = u.load_np('drugs')

drugID = drugs[i]

q = 'select count(case when atc_5_id = '+str(drugID)+' then 1 end) as drug from atc_5_patient_psm group by PID order by PID'

drug_feature = np.array(db.get_list(q)).reshape(u.NUM_PATIENTS, 1)

u.save_feature(drug_feature, i)