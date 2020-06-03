from collections import defaultdict
import numpy as np
import pandas as pd 
import feather 
import scipy.stats
from utils import Utils
from database import Database
from drug import Drug

u = Utils()
db = Database('Mimir from Munnin')
np.random.seed(u.RANDOM_STATE)

df_patients = u.load_df('df_patients')
sex_adr = db.get_list('select meddra_pt_id from gender_terms')
drugs = db.get_list('select atc_5_id from atc_5_name')

zero = [(x,y) for x,y in u.load_np('prr_zero')]
test = [(x,y) for x,y in u.load_np('prr_test')]

set_zero = set(zero)
set_test = set(test)

for drugID in drugs:
    
    for adrID in sex_adr: 
        
        x = (drugID, adrID)
        if x in set_zero: continue 
        if x in set_test: continue 
            
        q = "select * from (select PID from pt_patient where meddra_concept_id = "+str(adrID)+") as has_adr inner join (select PID from atc_5_patient where atc_5_id = "+str(drugID)+") as takes_drug using (PID)"
        if not db.get_list(q): 
            zero.append(x)
        else: 
            test.append(x)
            
    u.save_np(np.array(zero, dtype=tuple), 'prr_zero')
    u.save_np(np.array(test, dtype=tuple), 'prr_test')