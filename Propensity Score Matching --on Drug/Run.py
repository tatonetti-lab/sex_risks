#imports 
import multiprocessing
import datetime
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

        # before drug processed: read data, set date 
        
        today = datetime.date.today().isoformat()
        df_aeolus = feather.read_dataframe('Data/AEOLUS_clean_ranked.feather')

        # single drug workflow 
        def run(drug, today=today, df_aeolus=df_aeolus):

                drugID = df_aeolus.query('drug_concept_name==@drug').get('drug_concept_id').values.item(0)

                def complete(message, drug=drug):
                    LOG_PATH = "complete.txt".format(drug)
                    message = str(drug)+"\t"+str(message)+"\n"

                    if os.path.exists(LOG_PATH):
                            with open(LOG_PATH, 'a') as log:
                                    log.write(message)
                    else:
                            with open(LOG_PATH, 'w') as log:
                                    log.write(message)

                def addResult(message, newFile=False, drug=drug):
                    LOG_PATH = "{}/result.txt".format(drug)
                    message = str(message)+"\n"

                    if (os.path.exists(LOG_PATH) & (not newFile)):
                            with open(LOG_PATH, 'a') as log:
                                    log.write(message)
                    else:
                            with open(LOG_PATH, 'w') as log:
                                    log.write(message)

                def log(message, drug=drug):
                    LOG_PATH = "{}/log.txt".format(drug)
                    message = str(message)+"\n"

                    if os.path.exists(LOG_PATH):
                            with open(LOG_PATH, 'a') as log:
                                    log.write(message)
                    else:
                            with open(LOG_PATH, 'w') as log:
                                    log.write("\n" + "Created on " + today + "\n")
                                    log.write(message)

                def checkLog(message, drug=drug):
                    LOG_PATH = "{}/log.txt".format(drug)
                    message = str(message)

                    if not os.path.exists(LOG_PATH):
                        return False

                    if message in open(LOG_PATH, 'r').read():
                        return True
                    else:
                        return False
                
                # PSM functions : 
                def logPsm(message, drug=drug):
                        LOG_PATH = "{}/PSM/PSMlog.txt".format(drug)
                        message = str(message)+"\n"

                        if os.path.exists(LOG_PATH):
                                with open(LOG_PATH, 'a') as log:
                                        log.write(message)
                        else:
                                with open(LOG_PATH, 'w') as log:
                                        log.write("\n" + "\n" + "PSM started on " + today + "\n")
                                        log.write(message)

                def checkLogPsm(message, drug=drug):
                    LOG_PATH = "{}/PSM/PSMlog.txt".format(drug)
                    message = str(message)

                    if not os.path.exists(LOG_PATH):
                        return False

                    if message in open(LOG_PATH, 'r').read():
                        return True
                    else:
                        return False

                def makeHistogram(df, ylim=1000, drug=drug): # df with propensity scores
                    cases = df.query('{}==1'.format(drug))
                    controls = df.query('{}==0'.format(drug))
                    bins = np.linspace(0, 1, 100)
                    plt.hist(controls.Propensity, bins, alpha=0.5, label='controls')
                    plt.hist(cases.Propensity, bins, alpha=0.5, label='cases')
                    plt.ylim(0,ylim)
                    plt.title("LRCV Histogram: "+drug)
                    plt.xlabel("Propensity Score")
                    plt.legend(loc='upper right')
                    plt.savefig("{}/PSM/Histogram".format(drug))
                    logPsm("Hist made")
                    plt.close()

                def doROC(df):
                    y_true = (df.get(drug)>0).astype(np.int8).values
                    y_scores = df.get('Propensity').values
                    AUROC = roc_auc_score(y_true, y_scores)
                    # cross validated AUROC: 
                    # a, p, AUROC_CV = cross_val_score(model, X, y,scoring='roc_auc')
                    logPsm("AUROC: "+str(AUROC))
                    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                    data=pd.DataFrame({'fpr':fpr, 'tpr':tpr})
                    plt.plot('fpr','tpr', data=data, marker='o',ms=0.1)   
                    plt.title(drug+" (AUROC: "+str(round(AUROC,2))+")")
                    plt.xlabel("1 - Specificity")
                    plt.ylabel("Sensitivity")
                    plt.savefig("{}/PSM/ROC".format(drug))
                    logPsm("ROC made")
                    plt.close()

                def doPSM():

                    # Make new sub directory in drug folder directory if required

                    if not os.path.exists("{}/PSM".format(drug)):
                        os.mkdir("{}/PSM".format(drug))

                    # Make LR model & generate pscores if not previously saved
                    if not checkLogPsm("pscores saved"):
                        if checkLogPsm("Pickled"): # read previous model
                            pkl_filename = "{}/PSM/LRCV.pkl".format(drug)
                            with open(pkl_filename, 'rb') as file:  
                                propensity = pickle.load(file)

                            df_PSM_all = feather.read_dataframe('Data/Generic Matrix.feather').set_index('ID')
                            df_PSM_all.index.name = 'ID'

                            covariates = np.load("{}/PSM/covariates.npy".format(drug))
                            X = df_PSM_all.loc[:,covariates]

                        else: # generate new model
                            logPsm(drug)
                            #read generic PSM matrix 
                            df_PSM_all = feather.read_dataframe('Data/Generic Matrix.feather').set_index('ID')
                            df_PSM_all.index.name = 'ID'
                            logPsm("Read general PSM data")
                            
                            #getting COV (age, num drugs, concomittant > 1%)
                            cases = df_PSM_all.query('{}==1'.format(drug))
                            covariates = (cases>0).astype(np.int8).mean(axis=0).ge(0.01)
                            covariates = np.array(covariates.index)[np.array(covariates.values)].tolist()
                            covariates.remove(drug)
                            covariates.remove('Sex')
                            np.save("{}/PSM/covariates.npy".format(drug), covariates)
                            logPsm("Covariates written")

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
                            logPsm("Fitted LR model w CV, L2")

                            pkl_filename = "{}/PSM/LRCV.pkl".format(drug)
                            with open(pkl_filename, 'wb') as file:  
                                pickle.dump(propensity, file)
                            logPsm("Pickled")

                            retained = len([b for b in (np.abs(propensity.coef_[0])) if b > 0])
                            if (retained==len(covariates)):
                                logPsm("Covariates retained: ALL " + str(len(covariates)))
                            else:
                                logPsm("Covariates retained: "+str(retained)+" out of " +str(len(covariates)))

                        # generate propensity scores from model
                        pscore = propensity.predict_proba(X)[:,1] 
                        df_PSM_all['Propensity'] = pscore
                        logPsm("pscores predicted & assigned")

                        df_PSM_all.loc[:,[drug,'Propensity']].reset_index().to_feather("{}/PSM/LRCV pscores.feather".format(drug))
                        logPsm("pscores saved in dataframe")
                        log("PSM done")

                        makeHistogram(df_PSM_all)
                        doROC(df_PSM_all)

                # workflow functions:
                def match(df, cal=0.05, minCtrl = 10, numBins = 100):
                    allCases = df.copy(deep=True).query('{}==1'.format(drug))
                    allControls = df.query('{}==0'.format(drug))
                    
                    if (len(allCases)<50):
                        complete('insufficient cases')
                        quit()
                    
                    allCases['bin'] = pd.cut(allCases.Propensity, numBins)
                    mCases, mControls = [], []
                    
                    for cBin, caseData in allCases.groupby(by='bin'):
                        minPS, maxPS = cBin.left, cBin.right
                        numCtrl = len(caseData)*minCtrl
                        
                        controlOptions = allControls.query('Propensity > (@minPS-@cal) and Propensity < (@maxPS+@cal)').get('ID').values
                        
                        if (len(controlOptions)>minCtrl):
                            mCases = np.append(mCases, caseData.get('ID').values)
                            mControls = np.append(mControls, (np.random.choice(controlOptions, numCtrl)))
                    
                    return mCases, mControls

                def selectData(cases, controls, source=df_aeolus, drug=drug):
                    selection = []

                    selection.append(source[source.id.isin(cases)])

                    ctrl_unique, ctrl_counts = np.unique(controls, return_counts=True)

                    for count in np.unique(ctrl_counts):  
                        idx = np.where(ctrl_counts==count)
                        ctrls = np.take(ctrl_unique, idx)

                        toAdd = source[source.id.isin(np.hstack(ctrls))]
                        for i in range(0,count): 
                                selection.append(toAdd)

                    return pd.concat(selection, ignore_index=True)

                # Make new directory for drug if required
                if not os.path.exists("{}".format(drug)):
                    os.mkdir("{}".format(drug))

                log("\n\n"+drug+"  "+today)

                if(not checkLog("PSM done")):
                    doPSM()

                scores = feather.read_dataframe("{}/PSM/LRCV pscores.feather".format(drug))
                cases, controls = match(scores)
                df_matched = selectData(cases, controls)

                def doChiSquareX(cases, source=df_matched, drugName=drug, drugID=drugID):
                    chiResult = []
                    df_drug = source[source.id.isin(cases)].query('drug_concept_id==@drugID')

                    XM = df_drug.query('gender_code=="M"').shape[0]
                    XF = df_drug.query('gender_code=="F"').shape[0]

                    for outID in df_drug.get('snomed_outcome_concept_id').unique():

                        df_out = df_drug.query('snomed_outcome_concept_id==@outID')

                        outName = df_out.iloc[0].get('outcome_concept_name')
                        outRank = df_out.iloc[0].get('outcome_rank')

                        XME = df_out.query('gender_code=="M"').shape[0]
                        XFE = df_out.query('gender_code=="F"').shape[0]

                        if ((not XME == 0) and (not XFE == 0)):
                            contingencyTable = [[XFE, XF], [XME, XM]]
                        else:
                            contingencyTable = [[1, 1], [1, 1]]

                        chi2, p, dof, expected = sc.stats.chi2_contingency(contingencyTable)

                        cols = ['snomed_outcome_concept_id', 'outcome_concept_name', 'outcome_rank', 'p_valueX', 'XFE', 'XME', 'XM', 'XF']
                        data = [[outID, outName, outRank, float(p), XFE, XME, XM, XF]]

                        chiResult.append(pd.DataFrame(columns=cols, data=data))

                    return pd.concat(chiResult, ignore_index=True) 

                def bonferroni(df, colName):
                    p_values = pd.to_numeric(df[colName])
                    bonferroni_pvalues = p_values * len(p_values)
                    df = df.assign(newCol=bonferroni_pvalues)
                    newCol = 'bonf_' + colName
                    return df.rename(columns={'newCol': newCol})

                def getX(cases, alpha=0.05):

                    df_chi = doChiSquareX(cases)
                    df_bonf = bonferroni(df=df_chi, colName='p_valueX' )

                    return df_bonf.query('bonf_p_valueX<=@alpha')

                def doChiSquareE(df, source=df_matched, drugName=drug):
                    
                    TF = source.query('gender_code=="F"').shape[0]
                    TM = source.query('gender_code=="M"').shape[0]
                    
                    for idx, data in df.iterrows():
                        
                        df_out = source.query('snomed_outcome_concept_id==@data.snomed_outcome_concept_id')
                        
                        TFE = df_out.query('gender_code=="F"').shape[0]
                        TME = df_out.query('gender_code=="M"').shape[0]
                        
                        contingencyTable = [[TFE, TF], [TME, TM]]
                        chi2, p, dof, expected = sc.stats.chi2_contingency(contingencyTable)
                        
                        df.at[idx, 'p_valueE'] = p
                        df.at[idx, 'TFE'] = TFE
                        df.at[idx, 'TME'] = TME 
                        
                    return df

                def getE(df, alpha=0.05):
                    df_chi = doChiSquareE(df)
                    df_bonf = bonferroni(df=df_chi, colName='p_valueE' )
                    
                    df_bonf['p_valueE'] = df_bonf.get('p_valueE').mask(df_bonf.get('p_valueE')>alpha, float('nan'))
                    
                    return df_bonf

                def RORe(df, source=df_matched, drug=drug):
                    
                    TF = source.query('gender_code=="F"').shape[0]
                    TM = source.query('gender_code=="M"').shape[0]
                    
                    df['TFe'] = TF-df['TFE']
                    df['TMe'] = TM-df['TME']
                    
                    df.eval('RORE = log((TFE/TFe)/(TME/TMe))', inplace=True)
                    
                    return df

                def RORx(df): 
                    df['XFe'] = df['XF']-df['XFE']
                    df['XMe'] = df['XM']-df['XME']
                    
                    df.eval('RORX = log((XFE/XFe)/(XME/XMe))', inplace=True)
                    
                    return df

                def normalize(df, cols=['RORX','RORE']):
                    for col in cols: 
                        x = np.fabs(df[col])
                        x = (x-x.min())/(x.max()-x.min())
                        df = df.assign(xNorm=x).rename(columns={'xNorm': col+"N"})
                        
                    return df

                def getRisk(df):
                    
                    for i, data in df.iterrows():
                        e = data.RORE
                        x = data.RORX

                        if(data.p_valueE):
                            if(x<0):
                                if((e<0)&(x<e)):
                                    df.at[i, 'RORa'] = x-e
                            else:
                                if((e>0)&(e<x)):
                                    df.at[i, 'RORa'] = x-e
                                
                    df_risk = df
                    df_risk.query('RORX>1 | RORX<-1', inplace=True)
                    df_risk = normalize(df_risk, cols=['RORa']).eval('risk=RORaN*outcome_rank')
                    
                    for i, data in df_risk.iterrows():
                        if(data.RORa<0):
                            df_risk.at[i, 'risk'] = data.risk*(-1)
                    
                    return df_risk

                df_XE = getE(getX(cases))
                log("XE done")

                df_ROR = RORe(RORx(df_XE))
                log("ROR done")

                df_risk = getRisk(df_ROR)

                addResult(df_risk.get('risk').sum(), newFile=True)

                topRisks = "\nHighest Risks to Females: \n"+df_risk.nlargest(5, 'risk')\
                .get(['outcome_concept_name','risk','snomed_outcome_concept_id'])\
                .set_index(np.arange(1,6))\
                .query('risk>0')\
                .to_string(header=False)+"\n\nHighest Risks to Male: \n"+df_risk.nsmallest(5, 'risk')\
                .get(['outcome_concept_name','risk','snomed_outcome_concept_id'])\
                .set_index(np.arange(1,6))\
                .query('risk<0')\
                .to_string(header=False, justify='justify')

                addResult(topRisks)
                df_risk.reset_index().drop('index',axis=1).to_feather("{}/Risks.feather".format(drug))
                log("Risk done")

                complete("Success")
                return

        for drug in argv[1:]:
            run(drug)

if __name__ == "__main__":
        main(sys.argv)
