# AwareDX: Using machine learning to identify drugs posing increased risk of adverse reactions to women

### Summary

Adverse drug reactions (ADRs) are the fourth leading cause of death in the US. Although women take longer to metabolize medications and experience twice the risk of developing ADRs compared to men, these sex differences are not comprehensively understood. Real-world clinical data provides an opportunity to estimate safety effects in otherwise understudied populations, ie. women. These data, however, are subject to confounding biases and correlated covariates. We present AwareDX, a pharmacovigilance algorithm that leverages advances in machine learning to study sex risks. Our algorithm mitigates these biases and quantifies the differential risk of a drug causing an adverse event in either men or women. We present a resource of 20,817 adverse drug effects posing sex specific risks. We independently validated our algorithm against known pharmacogenetic mechanisms of genes that are sex-differentially expressed. AwareDX presents an opportunity to minimize adverse events by tailoring drug prescription and dosage to sex.


### Database requirements
To be able to run this project, it is necessary access to two databases, OpenFDA and AWAREdx.

#### OpenFDA
- drugs
- drugs_atc
- report
- patient

#### AWAREdx
- atc_4_name
- atc_5_name
- atc_4_name
- atc_5_patient
- atc_5_patient_psm
- pt_patient
- pt_name
- hglt_patient
- hlgt_name
- soc_patient
- soc_name

### Other requirements

It is necessary to have access to the following OMOP concept tables stores as CSV:
- CONCEPT
- CONCEPT_ANCESTOR
- CONCEPT_RELATIONSHIP

We need the following folder setup:
```
├── Code
├── Data
│   ├── Ad_Hoc
│   ├── PSM_features
│   ├── PSM_models
│   │   └── RF2
│   ├── Results
│   └── Status
└── Results
```

---

### Run

To run the code, it is necessary to install the requirements: ```pip install -r requirements.txt```
Then, we need to ensure we set the correct information in a config.ini file for the database connection.
Finally run: ```python3 Code/pipeline.py```
