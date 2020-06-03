import sys
from collections import Counter
import numpy as np
import pandas as pd 
import feather 
from scipy import stats
import pymysql
import pymysql.cursors
from database import Database
from utils import Utils
from drug import Drug

np.random.seed(222020)

def main(argv):
    
    u = Utils()
    iterations = 25
    
    idx = int(argv[1])
    drugs = u.load_np('drugs')
    drugID =  drugs[idx]
    
    status = u.read_status(drugID)

    if status=='no': 

        try:

            u.write_status(drugID, 'working')

            drug = Drug(drugID)

            for itr in range(1, iterations+1): 
                drug.match()
                drug.count_adr()
                drug.assign_abcd(itr)
                drug.do_chi_square()
                drug.calc_logROR()
                drug.reset_for_next_itr()

            x = drug.save_results(iterations)

            if x: 
                u.write_status(drugID, 'yes')
            else: 
                u.write_status(drugID, 'no')

        except:
            
            info = str(sys.exc_info()[1])
            u.write_status(drugID, 'error '+info)
            
if __name__ == "__main__":
        main(sys.argv)

