import pyarrow.feather as feather
from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np


def load_concept_table():
    CONCEPT_TABLE_FILE = 'CONCEPT.csv'
    concept_table = pd.read_csv(CONCEPT_TABLE_FILE, sep='\t')

    return concept_table


def load_concept_relationship_table():
    CONCEPT_RELATIONSHIP_TABLE_FILE = 'CONCEPT_RELATIONSHIP.csv'
    concept_relationship_table = pd.read_csv(CONCEPT_RELATIONSHIP_TABLE_FILE, sep='\t')

    return concept_relationship_table


def load_concept_ancestor_table():
    CONCEPT_ANCESTOR_TABLE_FILE = 'CONCEPT_ANCESTOR.csv'
    concept_ancestor_table = pd.read_csv(CONCEPT_ANCESTOR_TABLE_FILE, sep='\t')

    return concept_ancestor_table


def load_sex_risks():
    sex_risks = feather.read_table(
        './Data/sex_risks.feather')
    sex_risks = sex_risks.to_pandas()

    sex_risks['drug'] = pd.to_numeric(sex_risks['drug'])
    sex_risks['adr'] = pd.to_numeric(sex_risks['adr'])

    return sex_risks


def load_top_drugs():
    top_300_drugs_link = "https://clincalc.com/DrugStats/Top300Drugs.aspx"
    resp = requests.get(top_300_drugs_link)
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = soup.find(id="tableTopDrugs")
    drugs = []

    for trow in table.find_all("tr")[1:]:
        tds = trow.find_all('td')
        rank = int(tds[0].get_text())
        drug_name = tds[1].get_text()
        drug_name = str.lower(drug_name)

        drugs.append({'rank': rank, 'drug_name': drug_name})

    drugs = pd.DataFrame.from_records(drugs)

    return drugs


def get_top_adrs(meddra_concepts):
    # adr_file = "./top_adr.xlsx"
    adr_file = "./FAERS_ADE_Severity.csv"
    top_adrs = pd.read_csv(adr_file, header=0)

    # top_adrs = top_adrs.iloc[:n]
    top_adrs['Event'] = top_adrs['Event'].apply(str.lower)

    top_adrs = top_adrs.merge(meddra_concepts,
                              left_on=['Event'],
                              right_on='concept_name')

    return top_adrs


def filter_sex_risks_1():
    print("Loading concept table")
    concept_table = load_concept_table()
    print("Loading concept relationship table")
    concept_relationship_table = load_concept_relationship_table()
    concept_relationship_table = concept_relationship_table[
        concept_relationship_table['RELATIONSHIP_ID'].isin(['Is a',
                                                            'Maps to'])]

    print("Loading concept ancestor table")
    concept_ancestor = load_concept_ancestor_table()

    drug_concepts = concept_table[(concept_table['CONCEPT_CLASS_ID'].isin(
        ['Ingredient', 'Multiple Ingredients', 'Precise Ingredient'])) &
                                  (concept_table['VOCABULARY_ID'] == 'RxNorm')]
    drug_concepts['CONCEPT_NAME'] = drug_concepts['CONCEPT_NAME'].apply(
        str.lower)

    atc_concepts = concept_table[(concept_table['VOCABULARY_ID'] == 'ATC')]

    meddra_concepts = concept_table[(
        concept_table['VOCABULARY_ID'] == 'MedDRA')]
    meddra_concepts['CONCEPT_NAME'] = meddra_concepts['CONCEPT_NAME'].apply(
        str.lower)

    print("Loading sex risks")
    sex_risks = load_sex_risks()
    sex_risks_columns = list(sex_risks.columns)

    print("Loading top drugs")
    drugs = load_top_drugs()
    filtered_drugs = []
    for _, drug in drugs.iterrows():
        rank, drug_name = drug['rank'], drug['drug_name']

        if ';' not in drug_name:
            concept_name = drug_name
        else:
            concept_name = drug_name.replace("; ", " / ")

        concept = drug_concepts[(
            drug_concepts['CONCEPT_NAME'] == concept_name)]

        if len(concept) > 0:
            filtered_drugs.append({'rank': rank, 'CONCEPT_NAME': concept_name})

        else:
            print(f'unmatched: {concept_name}')

    filtered_drugs = pd.DataFrame.from_records(filtered_drugs)

    filtered_drugs = filtered_drugs.merge(drug_concepts, on=['CONCEPT_NAME'])

    filtered_drugs = atc_concepts.merge(concept_relationship_table[
        concept_relationship_table['RELATIONSHIP_ID'] == 'Maps to'],
                                        left_on=['CONCEPT_ID'],
                                        right_on=['CONCEPT_ID_1']).merge(
                                            filtered_drugs,
                                            left_on=['CONCEPT_ID_2'],
                                            right_on=['CONCEPT_ID'],
                                            suffixes=('_ATC', '_DRUG'))

    top_adrs = get_top_adrs(meddra_concepts)
    top_adrs['adr_rank'] = top_adrs.index
    top_adrs = top_adrs.rename(columns={'Event': 'top_adr_name'})

    sex_risks = sex_risks.merge(concept_ancestor,
                                left_on=['adr'],
                                right_on=['ANCESTOR_CONCEPT_ID']).merge(
                                    top_adrs,
                                    left_on=['DESCENDANT_CONCEPT_ID'],
                                    right_on=['CONCEPT_ID'])

    sex_risks = sex_risks.merge(filtered_drugs,
                                left_on=['drug'],
                                right_on=['CONCEPT_ID_ATC'])
    columns_to_include = set(sex_risks_columns) | set([
        'adr', 'adr_name', 'adr_class', 'top_adr_name', 'drug', 'drug_name',
        'drug_class', 'rank', 'adr_rank'
    ])
    sex_risks = sex_risks[columns_to_include]
    sex_risks = sex_risks.rename(
        columns={
            'rank': 'drug_rank',
            'XFE': 'female_report_count',
            'XME': 'male_report_count'
        })

    sex_risks = sex_risks.sort_values(by=['logROR_ci95_low'],
                                      ascending=[False])

    sex_risks = sex_risks[[
        'drug_rank', 'drug', 'drug_name', 'adr_rank', 'adr', 'adr_name',
        'logROR_avg', 'p_val_med', 'female_report_count', 'male_report_count'
    ]]
    sex_risks.to_csv('sex_risks_pt.csv', index=False)
    # print(sex_risks)
    # sex_risks = sex_risks[sex_risks['adr'].isin(top_adrs)]

    return sex_risks

def filter_sex_risks():
    print("Loading concept table")
    concept_table = load_concept_table()
    print("Loading concept relationship table")
    concept_relationship_table = load_concept_relationship_table()
    concept_relationship_table = concept_relationship_table[
        concept_relationship_table['relationship_id'].isin(['Is a',
                                                            'Maps to'])]

    print("Loading concept ancestor table")
    concept_ancestor = load_concept_ancestor_table()

    drug_concepts = concept_table[(concept_table['concept_class_id'].isin(
        ['Ingredient', 'Multiple Ingredients', 'Precise Ingredient'])) &
                                  (concept_table['vocabulary_id'] == 'RxNorm')]
    drug_concepts['concept_name'] = drug_concepts['concept_name'].apply(
        str.lower)

    atc_concepts = concept_table[(concept_table['vocabulary_id'] == 'ATC')]

    meddra_concepts = concept_table[(
        concept_table['vocabulary_id'] == 'MedDRA')]
    meddra_concepts['concept_name'] = meddra_concepts['concept_name'].apply(
        str.lower)

    print("Loading sex risks")
    sex_risks = load_sex_risks()
    sex_risks_columns = list(sex_risks.columns)

    print("Loading top drugs")
    drugs = load_top_drugs()
    filtered_drugs = []
    for _, drug in drugs.iterrows():
        rank, drug_name = drug['rank'], drug['drug_name']

        if ';' not in drug_name:
            concept_name = drug_name
        else:
            concept_name = drug_name.replace("; ", " / ")

        concept = drug_concepts[(
            drug_concepts['concept_name'] == concept_name)]

        if len(concept) > 0:
            filtered_drugs.append({'rank': rank, 'concept_name': concept_name})

        else:
            print(f'unmatched: {concept_name}')

    filtered_drugs = pd.DataFrame.from_records(filtered_drugs)

    filtered_drugs = filtered_drugs.merge(drug_concepts, on=['concept_name'])

    filtered_drugs = atc_concepts.merge(concept_relationship_table[
        concept_relationship_table['relationship_id'] == 'Maps to'],
                                        left_on=['concept_id'],
                                        right_on=['concept_id_1']).merge(
                                            filtered_drugs,
                                            left_on=['concept_id_2'],
                                            right_on=['concept_id'],
                                            suffixes=('_atc', '_drug'))

    top_adrs = get_top_adrs(meddra_concepts)
    top_adrs['adr_rank'] = top_adrs.index
    top_adrs = top_adrs.rename(columns={'Event': 'top_adr_name'})

    sex_risks = sex_risks.merge(concept_ancestor,
                                left_on=['adr'],
                                right_on=['ancestor_concept_id']).merge(
                                    top_adrs,
                                    left_on=['descendant_concept_id'],
                                    right_on=['concept_id'])

    sex_risks = sex_risks.merge(filtered_drugs,
                                left_on=['drug'],
                                right_on=['concept_id_atc'])
    columns_to_include = set(sex_risks_columns) | set([
        'adr', 'adr_name', 'adr_class', 'top_adr_name', 'drug', 'drug_name',
        'drug_class', 'rank', 'adr_rank'
    ])
    sex_risks = sex_risks[columns_to_include]
    sex_risks = sex_risks.rename(
        columns={
            'rank': 'drug_rank',
            'XFE': 'female_report_count',
            'XME': 'male_report_count'
        })

    sex_risks = sex_risks.sort_values(by=['logROR_ci95_low'],
                                      ascending=[False])

    sex_risks = sex_risks[[
        'drug_rank', 'drug', 'drug_name', 'adr_rank', 'adr', 'adr_name',
        'logROR_avg', 'p_val_med', 'female_report_count', 'male_report_count'
    ]]
    sex_risks.to_csv('sex_risks_pt.csv', index=False)
    # print(sex_risks)
    # sex_risks = sex_risks[sex_risks['adr'].isin(top_adrs)]

    return sex_risks


# def main2():
#     print("Loading concept table")
#     concept_table = load_concept_table()
#     print("Loading concept relationship table")
#     concept_relationship_table = load_concept_relationship_table()
#     concept_relationship_table = concept_relationship_table[
#         concept_relationship_table['relationship_id'].isin(['Is a',
#                                                             'Maps to'])]

#     drug_concepts = concept_table[(concept_table['concept_class_id'].isin(
#         ['Ingredient', 'Multiple Ingredients', 'Precise Ingredient'])) &
#                                   (concept_table['vocabulary_id'] == 'RxNorm')]
#     drug_concepts['concept_name'] = drug_concepts['concept_name'].apply(
#         str.lower)

#     atc_concepts = concept_table[(concept_table['vocabulary_id'] == 'ATC')]

#     print("Loading top drugs")
#     drugs = load_top_drugs()
#     filtered_drugs = []
#     for _, drug in drugs.iterrows():
#         rank, drug_name = drug['rank'], drug['drug_name']

#         if ';' not in drug_name:
#             concept_name = drug_name
#         else:
#             concept_name = drug_name.replace("; ", " / ")

#         concept = drug_concepts[(
#             drug_concepts['concept_name'] == concept_name)]

#         if len(concept) > 0:
#             filtered_drugs.append({'rank': rank, 'concept_name': concept_name})

#         else:
#             print(f'unmatched: {concept_name}')

#     filtered_drugs = pd.DataFrame.from_records(filtered_drugs)

#     filtered_drugs = filtered_drugs.merge(drug_concepts, on=['concept_name'])

#     filtered_drugs = atc_concepts.merge(concept_relationship_table[
#         concept_relationship_table['relationship_id'] == 'Maps to'],
#                                         left_on=['concept_id'],
#                                         right_on=['concept_id_1']).merge(
#                                             filtered_drugs,
#                                             left_on=['concept_id_2'],
#                                             right_on=['concept_id'],
#                                             suffixes=('_atc', '_drug'))

#     filtered_drugs = filtered_drugs[[
#         'concept_id_atc', 'concept_name_atc', 'rank'
#     ]]
#     filtered_drugs = filtered_drugs.rename(columns={
#         'concept_id_atc': 'concept_id',
#         'concept_name_atc': ' concept_name'
#     })

#     filtered_drugs.to_csv('top_drugs_with_concepts_ids.csv', index=False)

# def main3():
#     print("Loading concept table")
#     concept_table = load_concept_table()

#     meddra_concepts = concept_table[(
#         concept_table['vocabulary_id'] == 'MedDRA')]
#     meddra_concepts['concept_name'] = meddra_concepts['concept_name'].apply(
#         str.lower)

#     top_adrs = get_top_adrs(meddra_concepts, n=50)

#     top_adrs = top_adrs[['concept_id', 'concept_name']]
#     top_adrs['rank'] = top_adrs.index
#     top_adrs.to_csv('top_adrs_with_concepts_ids.csv', index=False)

if __name__ == '__main__':
    filter_sex_risks()

