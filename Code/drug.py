from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from scipy import stats
import pymysql
import pymysql.cursors
from database import Database
from utils import Utils
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

class Drug:

    u = Utils()

    def __init__(self, drugID, adrID=None):

        self.db = Database('awaredx')
        self.id = str(drugID)
        self.atc = self._get_atc()
        self.name = self._get_name()
        if adrID:
            self.adr = adrID
        else:
            self.adr = self._get_adr()
        self.pid = self._get_patients_on_drug()
        self.pscores = self._get_propensity_scores()
        self.results = self._get_blank_results()

        # self.results['XF'] = self.results['XF'].astype(int)
        self.results['XFE'] = self.results['XFE'].astype(int)
        self.results['XME'] = self.results['XME'].astype(int)

        self.match_f = None
        self.match_m = None
        self.XF = None
        self.XM = None
        self.adr_count_f = None
        self.adr_count_m = None

    def _get_atc(self):
        q_atc5 = 'select * from atc_5_name where atc_5_id=' + self.id
        q_atc4 = 'select * from atc_4_name where atc_4_id=' + self.id
        if self.db.get_list(q_atc4): atc = '4'
        elif self.db.get_list(q_atc5): atc = '5'
        else: raise NameError('Cant find drug in database')
        return atc

    def _get_name(self):
        q = 'select atc_' + self.atc + '_name from atc_' + self.atc + '_name where atc_' + self.atc + '_id=' + self.id
        name = self.db.get_list(q)[0]
        return name

    def _get_adr(self):
        return self.u.load_np('adr')

    def _get_patients_on_drug(self):
        q = 'select PID from atc_' + self.atc + '_patient where atc_' + self.atc + '_id = ' + self.id
        pid = self.db.get_list(q)

        #restrict to patients 16yo and older

        return pid

    def _get_propensity_scores(self):
        df_propensity = self.u.load_df('df_propensity')
        df_propensity = df_propensity.set_index('PID')
        return df_propensity[df_propensity.index.isin([str(i) for i in self.pid])]

    def _get_blank_results(self):
        df = pd.DataFrame(columns=[
            'drug', 'itr', 'adr', 'XFE', 'XFe', 'XME', 'XMe', 'p_value',
            'logROR'
        ])
        df = df.set_index(['drug', 'itr', 'adr'])
        return df

    def match(self, cal=0.01, bins=100, minRecPerBin=0.005):
        allFemales = self.pscores.copy(deep=True).query('Sex=="F"')
        allMales = self.pscores.copy(deep=True).query('Sex=="M"')

        if (len(allFemales) < 250) or (len(allMales) < 250):
            # print(len(allFemales), "females")
            # print(len(allMales), "males")
            raise NameError('Insufficient data for both sexes')

        minF = len(allFemales) * minRecPerBin
        minM = len(allMales) * minRecPerBin
        females, males = [], []
        allFemales['bin'] = pd.cut(allFemales.Propensity, bins)

        for iBin, fOptions in allFemales.groupby(by='bin'):
            minPS, maxPS = iBin.left, iBin.right

            mOptions = allMales.query(
                'Propensity > (@minPS-@cal) and Propensity < (@maxPS+@cal)'
            ).index.values

            if (len(mOptions) < minM or len(fOptions) < minF):
                continue
                # too few data, don't add to matched

            females = np.append(females, fOptions.index.values)
            males = np.append(males,
                              (np.random.choice(mOptions, len(fOptions))))

        self.match_f = females
        self.match_m = males

        self.XF = len(females)
        self.XM = len(males)

    def count_adr(self):

        # Females

        # q = 'select meddra_concept_id from hlgt_patient where PID in ("'+ "\", \"".join(self.match_f) + '")'
        # hglt_f = self.db.get_list(q)

        # q = 'select meddra_concept_id from soc_patient where PID in ("'+ "\", \"".join(self.match_f) + '")'
        # soc_f = self.db.get_list(q)

        # q = f'''
        # select meddra_concept_id from pt_patient where PID in ({", ".join([f'"{w}"' for w in self.match_f])})
        #     and meddra_concept_id in (35204966, 35205180, 35809059, 36110649, 36416501, 36717998, 36919230, 36919236, 37080784)
        # '''
        q = f'''
        select meddra_concept_id from pt_patient where PID in ({", ".join([f'"{w}"' for w in self.match_f])})
        '''
        pt_f = self.db.get_list(q)

        # select_f = hglt_f
        # select_f.extend(soc_f)
        # select_f.extend(pt_f)

        select_f = pt_f

        # Male

        select_m = []

        unique_m, counts = np.unique(self.match_m, return_counts=True)

        for count in np.unique(counts):
            idx = np.where(counts == count)
            pids = np.take(unique_m, idx)[0]

            # q = 'select meddra_concept_id from hlgt_patient where PID in ("'+ "\", \"".join(pids) + '")'
            # to_add_hlgt = self.db.get_list(q)

            # q = 'select meddra_concept_id from soc_patient where PID in ("'+ "\", \"".join(pids) + '")'
            # to_add_soc = self.db.get_list(q)

            # q = f'''
            #     select meddra_concept_id from pt_patient where PID in ({", ".join([f'"{w}"' for w in pids])})
            #         and meddra_concept_id in (35204966, 35205180, 35809059, 36110649, 36416501, 36717998, 36919230, 36919236, 37080784)
            #     '''
            q = f'''
                select meddra_concept_id from pt_patient where PID in ({", ".join([f'"{w}"' for w in pids])})
                '''
            to_add_pt = self.db.get_list(q)

            for i in range(count):
                # select_m.extend(to_add_hlgt)
                # select_m.extend(to_add_soc)
                select_m.extend(to_add_pt)

        self.adr_count_f = Counter(select_f)
        self.adr_count_m = Counter(select_m)

    def assign_abcd(self, itr):

        for adr in self.adr:
            self.results.loc[(self.id, itr, adr),
                             ['XFE']] = self.adr_count_f[adr]
            self.results.loc[(self.id, itr, adr),
                             ['XME']] = self.adr_count_m[adr]

        self.results.eval('XFe = @self.XF - XFE', inplace=True)
        self.results.eval('XMe = @self.XM - XME', inplace=True)

    def do_chi_square(self):

        for idx, data in self.results.iterrows():

            contingencyTable = np.array([[data.XFE, data.XFe],
                                         [data.XME, data.XMe]])

            if contingencyTable.all():
                chi2, p, dof, expected = stats.chi2_contingency(
                    contingencyTable)
                self.results.at[idx, 'p_value'] = p

    def calc_logROR(self):
        self.results.eval('ROR = (XFE/XFe)/(XME/XMe)', inplace=True)
        self.results.eval('logROR = log(ROR)', inplace=True)
        self.results.drop('ROR', axis=1, inplace=True)

    def reset_for_next_itr(self):
        self.match_f = None
        self.match_m = None
        self.XF = None
        self.XM = None
        self.adr_count_f = None
        self.adr_count_m = None

    def save_results(self, itr):
        assert self.ensure_results(itr)
        assert self.u.save_df(self.results, 'Results/' + self.id)
        return True

    def ensure_results(self, itr):
        self.results = self.results.reset_index()
        assert self.results.shape[0] == itr * len(self.adr)
        assert self.results.shape[1] == 9
        return True