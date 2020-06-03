from collections import defaultdict
import numpy as np
import pandas as pd 
import feather 

from utils import Utils
from database import Database

u = Utils()
db = Database('Mimir from Munnin')
np.random.seed(u.RANDOM_STATE)

# Check progress 

progress = defaultdict(int)
complete = []

drugs = u.load_np('drugs')

for drugID in drugs: 
    s = u.read_status(drugID)
    if s == 'yes': 
        complete.append(drugID)
    progress[s] += 1

for key in progress: 
    percentage = round((progress[key]/len(drugs))*100, 1)
    print(key, "\t", percentage, "%")   
    
# Compile results 

res = []
for drugID in complete: 
    r = u.load_df('Results/'+str(drugID))
    res.append(r)   
assert len(complete) == len(res)

results = pd.concat(res, ignore_index=True)
assert (results.shape[0] / (25 * len(complete))) == len(u.load_np('adr'))
u.save_df(results, 'results')

print("no of drugs: ", len( u.load_np('drugs')))
print("No of drug-adr pairs tested: ",results.get(['drug','adr']).drop_duplicates().shape[0])