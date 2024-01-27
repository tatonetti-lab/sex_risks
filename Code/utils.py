import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy.sparse import coo_matrix, save_npz, load_npz
import configparser

config = configparser.ConfigParser()
config.read('config.ini')


class Utils:

    def __init__(self):
        self.DATA_PATH = config['FILES']['data_path']
        self.RANDOM_STATE = 222020
        self.NUM_PATIENTS = 8860677

    def save_df(self, df, name):
        df.reset_index().drop('index', axis=1).to_feather(self.DATA_PATH +
                                                          name + ".feather")
        # print("Saved", name)
        return True

    def load_df(self, name):
        return pd.read_feather(self.DATA_PATH + name + ".feather")

    def save_np(self, arr, name):
        np.save(self.DATA_PATH + name + ".npy", arr)
        # print("Saved", name)

    def load_np(self, name):
        return np.load(self.DATA_PATH + name + ".npy", allow_pickle=True)

    def save_feature(self, data, name):
        save_npz(self.DATA_PATH + 'PSM_features/' + str(name) + '.npz',
                 coo_matrix(data))

    def load_feature(self, name):
        return load_npz(self.DATA_PATH + 'PSM_features/' + str(name) + '.npz')

    def read_status(self, drugID):
        filename = self.DATA_PATH + 'Status/' + str(drugID) + '.txt'
        with open(filename, 'r') as f:
            data = f.readlines()
        return data[0]

    def write_status(self, drugID, msg):
        filename = self.DATA_PATH + 'Status/' + str(drugID) + '.txt'
        with open(filename, 'w') as f:
            f.write(msg)
        return

    def show(self, df):
        with pd.option_context('display.max_rows', 50000,
                               'display.max_columns', 100):
            display(df)

    def print_table(self, df):
        for _, info in df.iterrows():
            print()
            if info.logROR_avg > 0:
                print(
                    "{sex} \t {ror:.2f} ({low:.2f}, {upp:.2f}) \t {drug} \t {level} {name}"
                    .format(sex='F',
                            level=info.adr_class,
                            name=info.adr_name,
                            ror=info.logROR_avg,
                            upp=info.logROR_ci95_upp,
                            low=info.logROR_ci95_low,
                            drug=info.drug_name))
            else:
                print(
                    "{sex} \t {ror:.2f} ({upp:.2f}, {low:.2f}) \t {drug} \t {level} {name}"
                    .format(sex='M',
                            level=info.adr_class,
                            name=info.adr_name,
                            ror=abs(info.logROR_avg),
                            upp=abs(info.logROR_ci95_upp),
                            low=abs(info.logROR_ci95_low),
                            drug=info.drug_name))

    def print_tex(self, df):
        for _, info in df.iterrows():
            print()
            if info.logROR_avg > 0:
                print(
                    "{drug} & .. & .. & {name} & {sex} & {ror:.2f} & ({low:.2f}, {upp:.2f}) \\\ "
                    .format(sex='F',
                            name=info.adr_name,
                            ror=info.logROR_avg,
                            upp=info.logROR_ci95_upp,
                            low=info.logROR_ci95_low,
                            drug=info.drug_name))
            else:
                print(
                    "{drug} & .. & .. & {name} & {sex} & {ror:.2f} & ({low:.2f}, {upp:.2f}) \\\ "
                    .format(sex='M',
                            name=info.adr_name,
                            ror=abs(info.logROR_avg),
                            upp=abs(info.logROR_ci95_upp),
                            low=abs(info.logROR_ci95_low),
                            drug=info.drug_name))


'''
Usage 

from utils import Utils
u = Utils()
'''