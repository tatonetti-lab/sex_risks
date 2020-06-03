import sys
import os 
from collections import defaultdict
import numpy as np
import pandas as pd 
import feather 
import scipy.stats
from utils import Utils
from database import Database
from drug import Drug


def main(argv):

    i=int(argv[1])

    u = Utils()
    db = Database('Mimir from Munnin')
    np.random.seed(u.RANDOM_STATE)

    df_patients = u.load_df('df_patients')
    sex_adr = db.get_list('select meddra_pt_id from gender_terms')
    drugs = db.get_list('select atc_5_id from atc_5_name')
    test = [(x,y) for x,y in u.load_np('prr_test')]

    df_test = pd.DataFrame(test, columns=['drug','adr']).sort_values(by='drug')[i:]

    PID_M = set(df_patients.query('Sex=="M"').get('PID').values)
    PID_F = set(df_patients.query('Sex=="F"').get('PID').values)

    for drugID, data in df_test.groupby('drug'): 

        filename = 'Post_PRR/'+str(drugID)
        pth = u.DATA_PATH+filename+'.feather'
        if os.path.exists(pth): 
            print(drugID)
            continue

        prr_counts = []

        q = 'select PID from atc_5_patient where atc_5_id = '+str(drugID)
        takes_drug = set(db.get_list(q))

        try: 
            drug = Drug(drugID)
            drug.match()
        except NameError:
            df = pd.DataFrame(columns=['drug', 'adr', 'sex', 'a_post', 'c_post'])
            u.save_df(df, filename)
            
        if drug.match_m is None or drug.match_f is None: 
            df = pd.DataFrame(columns=['drug', 'adr', 'sex', 'a_post', 'c_post'])
            u.save_df(df, filename)
            db = Database('Mimir from Munnin')
            continue 

        for adrID in data.adr.values: 
            q = 'select PID from pt_patient where meddra_concept_id = '+str(adrID)
            has_adr = set(db.get_list(q))

            for sex in ['M','F']: 
                if sex=='M': PSM = drug.match_m
                else: PSM = drug.match_f

                a_post = len([1 for x in PSM if x in (has_adr & takes_drug)])
                c_post = len([1 for x in PSM if x in (takes_drug - has_adr)])

                res = {'drug':drugID, 'adr':adrID, 'sex':sex, 'a_post':a_post, 'c_post':c_post}
                prr_counts.append(res)

        df = pd.DataFrame(prr_counts)
        u.save_df(df, filename)
        break
        
if __name__ == "__main__":
        main(sys.argv)
