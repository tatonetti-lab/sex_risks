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

results = u.load_df('results')

results = results.dropna()

num_tests = results.shape[0]
results.loc[:,'bonf_p_value'] = results.get('p_value') * num_tests
#results = results.query('bonf_p_value<1')

# 348285 pairs --> 302140 pairs 
drug_adr_pairs = results.get(['drug','itr','adr']).groupby(by=['drug','adr']).count().query('itr==25').reset_index().get(['drug', 'adr'])

scores = pd.DataFrame(columns=['drug', 'adr', 'p_val_min', 'p_val_med', 'p_val_max', 'logROR_avg','logROR_ci95_low', 'logROR_ci95_upp']).set_index(['drug','adr'])

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

for _, (drug, adr) in drug_adr_pairs.iterrows():
    data = results.query('drug==@drug and adr==@adr')
    
    bonf_p = data['bonf_p_value'].values 
    scores.at[(drug, adr), 'p_val_min'] = np.min(bonf_p)
    scores.at[(drug, adr), 'p_val_med'] = np.median(bonf_p)
    scores.at[(drug, adr), 'p_val_max'] = np.max(bonf_p)

    logROR = data['logROR'].values 
    mean, lower, upper = mean_confidence_interval(logROR)
    scores.at[(drug, adr), 'logROR_avg'] = mean
    scores.at[(drug, adr), 'logROR_ci95_low'] = lower
    scores.at[(drug, adr), 'logROR_ci95_upp'] = upper
    
name_atc4, name_atc5, name_hlgt, name_soc = defaultdict(str), defaultdict(str),  defaultdict(str),  defaultdict(str)

for id_, name in db.run('select * from atc_4_name'): 
    name_atc4[str(id_)] = name
    
for id_, name in db.run('select * from atc_5_name'): 
    name_atc5[str(id_)] = name

for id_, name in db.run('select * from hlgt_name'): 
    name_hlgt[id_] = name

for id_, name in db.run('select * from soc_name'): 
    name_soc[id_] = name
    
scores['drug_name'] = ''
scores['drug_class'] = 0
scores = scores.set_index('drug')
for id_ in np.unique(scores.index): 
    if name_atc4[id_]: 
        scores.at[id_, 'drug_name'] = name_atc4[id_]
        scores.at[id_, 'drug_class'] = 4
    else:
        scores.at[id_, 'drug_name'] = name_atc5[id_]
        scores.at[id_, 'drug_class'] = 5
scores = scores.reset_index()

scores['adr_name'] = ''
scores['adr_class'] = ''
scores = scores.set_index('adr')
for id_ in np.unique(scores.index): 
    if name_soc[id_]: 
        scores.at[id_, 'adr_name'] = name_soc[id_]
        scores.at[id_, 'adr_class'] = 'soc'
    else:
        scores.at[id_, 'adr_name'] = name_hlgt[id_]
        scores.at[id_, 'adr_class'] = 'hlgt'
scores = scores.reset_index()
    
u.save_df(scores.reset_index(), "compiled")

compiled = u.load_df('compiled')
sex_adrs = u.load_np('adr_hide')
sig_results = compiled.query('p_val_max<0.05').query('adr_name not in @sex_adrs')
u.save_df(sig_results "sex_risks")
