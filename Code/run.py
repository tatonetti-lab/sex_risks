import sys
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import stats
import pymysql
import pymysql.cursors
from database import Database
from utils import Utils
from drug import Drug
import tools

# from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

np.random.seed(222020)

u = Utils()
ITERATIONS = 25


def run_analysis(drugID):
    # try:

    # print(f'DrugID: {drugID}')

    # print('Reading status')
    # status = u.read_status(drugID)

    # if status == 'no':

    try:

        u.write_status(drugID, 'working')
        drug = Drug(drugID)

        #print('Iterating and matching')
        for itr in range(1, ITERATIONS + 1):
            drug.match()
            drug.count_adr()
            drug.assign_abcd(itr)
            drug.do_chi_square()
            drug.calc_logROR()
            drug.reset_for_next_itr()

        #print('Saving results')
        x = drug.save_results(ITERATIONS)

        if x:
            u.write_status(drugID, 'yes')
        else:
            u.write_status(drugID, 'no')

    except:
    #     # print('Failed miserably')
         info = str(sys.exc_info()[1])
         u.write_status(drugID, 'error ' + info)


def run_analysis_mp():
    print('Loading drugs')
    drugs = u.load_np('drugs')
    # for drug in drugs:
    #      run_analysis(drug)
    process_map(run_analysis, drugs, max_workers=35)


if __name__ == "__main__":
    run_analysis_mp()
