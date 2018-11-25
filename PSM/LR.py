#imports 
import multiprocessing
import feather
import matplotlib as mpt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
import sys
import pickle
import os
import shutil
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def main(argv):

        drug = argv

        # writes message to log file
        def log(message, drug=drug):
                LOG_PATH = "{}/log.txt".format(drug)
                message = str(message)+"\n"

                if os.path.exists(LOG_PATH):
                        with open(LOG_PATH, 'a') as log:
                                log.write(message)
                else:
                        with open(LOG_PATH, 'w') as log:
                                log.write("New log created\n")
                                log.write(message)

        # Make new drug directory to store results 
        if os.path.exists(drug):
                shutil.rmtree(drug)
        os.mkdir(drug)

        log(drug)
        #read generic PSM matrix 
        df_PSM_all = feather.read_dataframe('Generic Matrix.feather').set_index('ID')
        df_PSM_all.index.name = 'ID'
        log("Read general PSM data")
        #getting df_PSM for case (exposed to drug) & control (unexposed) cohorts
        cases = df_PSM_all.query('{}==1'.format(drug))
        controls = df_PSM_all.query('{}==0'.format(drug))

        #getting COV (age, num drugs, concomittant > 1%)
        covariates = (cases>0).astype(np.int8).mean(axis=0).ge(0.01)
        covariates = np.array(covariates.index)[np.array(covariates.values)].tolist()
        covariates.remove(drug)
        covariates.remove('Sex')
        log("Got Covariates: "+str(covariates))

        # setting x & y for logistic regression
        # x: covariates 
        # y: drug
        y = df_PSM_all.loc[:,[drug]]
        y=y.values.reshape(1,-1)[0]
        X = df_PSM_all.loc[:,covariates]

        # making logistic regression model
        # with cross validation, l2 penalty

        model = LogisticRegressionCV(penalty='l2')

        propensity = model.fit(X,y)
        log("Fitted LR model w CV, L2")

        pkl_filename = "{}/LRCV.pkl".format(drug)
        with open(pkl_filename, 'wb') as file:  
            pickle.dump(propensity, file)
        log("Pickled")

        retained = len([b for b in (np.abs(propensity.coef_[0])) if b > 0])
        if (retained==len(covariates)):
                log("Covariates retained: ALL " + str(len(covariates)))
        else:
                log("Covariates retained: "+str(retained)+" out of " +str(len(covariates)))

        pscore = propensity.predict_proba(X)[:,1] 
        df_PSM_all['Propensity'] = pscore
        log("pscores predicted & assigned")

        df_PSM_all.loc[:,[drug,'Propensity']].reset_index().to_feather("{}/LRCV pscores.feather".format(drug))
        log("pscores saved in dataframe")

        # cases/controls with propensity
        cases = df_PSM_all.query('{}==1'.format(drug))
        controls = df_PSM_all.query('{}==0'.format(drug))

        # hist 
        bins = np.linspace(0, 1, 100)
        plt.hist(controls.Propensity, bins, alpha=0.5, label='controls')
        plt.hist(cases.Propensity, bins, alpha=0.5, label='cases')
        plt.ylim(0,500)
        plt.title("LRCV Histogram: "+drug)
        plt.xlabel("Propensity Score")
        plt.legend(loc='upper right')
        plt.savefig("{}/Histogram".format(drug))
        log("Hist made")
        plt.close()

        # METRICS AUROC
        y_true = (df_PSM_all.get(drug)>0).astype(np.int8).values
        y_scores = df_PSM_all.get('Propensity').values
        AUROC = roc_auc_score(y_true, y_scores)
        log("AUROC: "+str(AUROC))

        # CV AUROC
        # a, p, AUROC_CV = cross_val_score(model, X, y,scoring='roc_auc')
        # log("AUROC from CV: "+str(AUROC_CV))

        # Plot ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        data=pd.DataFrame({'fpr':fpr, 'tpr':tpr})
        plt.plot('fpr','tpr', data=data, marker='o',ms=0.1)   
        plt.title(drug+" (AUROC: "+str(round(AUROC,2))+")")
        plt.xlabel("1 - Specificity")
        plt.ylabel("Sensitivity")
        plt.savefig("{}/ROC".format(drug))
        log("ROC made")
        plt.close()

if __name__ == "__main__":
        main(sys.argv[1])
