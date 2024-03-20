# AwareDX: Using machine learning to identify drugs posing increased risk of adverse reactions to women

### Summary

Adverse drug reactions (ADRs) are the fourth leading cause of death in the US. Although women take longer to metabolize medications and experience twice the risk of developing ADRs compared to men, these sex differences are not comprehensively understood. Real-world clinical data provides an opportunity to estimate safety effects in otherwise understudied populations, ie. women. These data, however, are subject to confounding biases and correlated covariates. We present AwareDX, a pharmacovigilance algorithm that leverages advances in machine learning to study sex risks. Our algorithm mitigates these biases and quantifies the differential risk of a drug causing an adverse event in either men or women. We present a resource of 20,817 adverse drug effects posing sex specific risks. We independently validated our algorithm against known pharmacogenetic mechanisms of genes that are sex-differentially expressed. AwareDX presents an opportunity to minimize adverse events by tailoring drug prescription and dosage to sex.


### Database requirements
To be able to run this project, it is necessary access to two databases, OpenFDA and AWAREdx.

#### OpenFDA
OpenFDA tables are created using the following repository https://github.com/ngiangre/openFDA_drug_event_parsing.
- drugcharacteristics
- drugs
- drugs_atc
- patient
- reactions
- report
- report_serious
- reporter
- standard_drugs
- standard_drugs_atc: in order to create this table it is necessary an extra mapping from RxNorm - atc
- standard_drugs_rxnorm_ingredients
- standard_reactions
- standard_reactions_meddra_hlgt
- standard_reactions_meddra_hlt
- standard_reactions_meddra_relationships
- standard_reactions_meddra_soc
- standard_reactions_snomed


#### AWAREdx
This tables are subsets from the OpenFDA database combined with additional information like CONCEP table from OMOP data structure. atc 4and 5 are extracted from CONCEPT table.
- atc_4_name
  - atc_5_id (concept_id)
  - atc_name
- atc_5_name
  - atc_5_id (concept_id)
  - atc_name
- atc_5_patient
  - PID (safetyreportid)
  - atc_5_id
- pt_patient
  - PID
  - meddra_concept_id
- pt_name
  - meddra_concept_id
  - meddra_concept_name
- hglt_patient
  - PID
  - meddra_concept_id
- hlgt_name
  - meddra_concept_id
  - meddra_concept_name
- soc_patient
  - PID
  - meddra_concept_id
- soc_name
  - meddra_concept_id
  - meddra_concept_name

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
