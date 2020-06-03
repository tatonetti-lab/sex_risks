from tqdm.notebook import tqdm
from collections import defaultdict
import numpy as np
import pandas as pd 
import feather 
import scipy.stats
from utils import Utils
from database import Database

u = Utils()
db = Database('Mimir from Munnin')
np.random.seed(u.RANDOM_STATE)

df_patients = u.load_df('df_patients')
sex_adr = db.get_list('select meddra_pt_id from gender_terms')
drugs = db.get_list('select atc_5_id from atc_5_name')
test = [(x,y) for x,y in u.load_np('prr_test')]

df_test = pd.DataFrame(test, columns=['drug','adr'])

PID_M = set(df_patients.query('Sex=="M"').get('PID').values)
PID_F = set(df_patients.query('Sex=="F"').get('PID').values)

prr_counts = []

for drugID, data in tqdm(df_test.groupby('drug')): 
    q = 'select PID from atc_5_patient where atc_5_id = '+str(drugID)
    takes_drug = set(db.get_list(q))
    
    try: 
        drug = Drug(drugID)
        drug.match()
        psm_done = True
    except NameError:
        psm_done = False
    
    for adrID in data.adr.values: 
        q = 'select PID from pt_patient where meddra_concept_id = '+str(adrID)
        has_adr = set(db.get_list(q))
        
        for sex in ['M','F']: 
            if sex=='M': PID = PID_M
            else: PID = PID_F
                
            a = len(PID & (has_adr & takes_drug))
            b = len(PID & (has_adr - takes_drug))
            c = len(PID & (takes_drug - has_adr))
            d = len(PID - (has_adr & takes_drug))
            
            if psm_done: 
                if sex=='M': PSM = drug.match_m
                else: PSM = drug.match_f
                a_post = len([1 for x in PSM if x in (has_adr & takes_drug)])
                c_post = len([1 for x in PSM if x in (takes_drug - has_adr)])
            else: 
                a_post = -1
                c_post = -1
            
            res = {'drug':drugID, 'adr':adrID, 'sex':sex, 'a':a, 'b':b, 'c':c, 'd':d, 'a_post':a_post, 'c_post':c_post}
            prr_counts.append(res)


df_prr_counts = pd.DataFrame(prr_counts)
u.save_df(df_prr_counts, 'df_prr_counts')            