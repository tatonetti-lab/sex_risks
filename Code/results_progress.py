from collections import defaultdict 
import numpy as np
from utils import Utils

u = Utils()
drugs = u.load_np('drugs')
progress = defaultdict(int)

for drugID in drugs: 
    s = u.read_status(drugID)
    s = s[:3]
    progress[s] += 1

for key in progress: 
    percentage = round((progress[key]/len(drugs))*100, 1)
    print(key, "\t", percentage, "%")