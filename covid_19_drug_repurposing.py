# -*- coding: utf-8 -*-
"""COVID-19_drug_repurposing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/142wdLLgNQQy6K1CSgTRTQmWZ3Ig74LHG

# COVID-19 Drug Repurposing via disease-compounds relations
This example shows how to do drug repurposing using DRKG even with the pretrained model.

## Collecting COVID-19 related disease
At the very beginning we need to collect a list of disease of Corona-Virus(COV) in DRKG. We can easily use the Disease ID that DRKG uses for encoding the disease. Here we take all of the COV disease as target.
"""

COV_disease_list = [
'Disease::SARS-CoV2 E',
'Disease::SARS-CoV2 M',
'Disease::SARS-CoV2 N',
'Disease::SARS-CoV2 Spike',
'Disease::SARS-CoV2 nsp1',
'Disease::SARS-CoV2 nsp10',
'Disease::SARS-CoV2 nsp11',
'Disease::SARS-CoV2 nsp12',
'Disease::SARS-CoV2 nsp13',
'Disease::SARS-CoV2 nsp14',
'Disease::SARS-CoV2 nsp15',
'Disease::SARS-CoV2 nsp2',
'Disease::SARS-CoV2 nsp4',
'Disease::SARS-CoV2 nsp5',
'Disease::SARS-CoV2 nsp5_C145A',
'Disease::SARS-CoV2 nsp6',
'Disease::SARS-CoV2 nsp7',
'Disease::SARS-CoV2 nsp8',
'Disease::SARS-CoV2 nsp9',
'Disease::SARS-CoV2 orf10',
'Disease::SARS-CoV2 orf3a',
'Disease::SARS-CoV2 orf3b',
'Disease::SARS-CoV2 orf6',
'Disease::SARS-CoV2 orf7a',
'Disease::SARS-CoV2 orf8',
'Disease::SARS-CoV2 orf9b',
'Disease::SARS-CoV2 orf9c',
'Disease::MESH:D045169',
'Disease::MESH:D045473',
'Disease::MESH:D001351',
'Disease::MESH:D065207',
'Disease::MESH:D028941',
'Disease::MESH:D058957',
'Disease::MESH:D006517'
]

"""## Candidate drugs
Now we use FDA-approved drugs in Drugbank as candidate drugs. (we exclude drugs with molecule weight < 250) The drug list is in infer\_drug.tsv
"""

import csv

# Load entity file
drug_list = []
with open("/Users/stevensmith/Projects/DRKG/drug_repurpose/infer_drug.tsv", newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['drug','ids'])
    for row_val in reader:
        drug_list.append(row_val['drug'])

len(drug_list)

"""## Treatment relation

Two treatment relations in this context
"""

treatment = ['Hetionet::CtD::Compound:Disease','GNBR::T::Compound:Disease']

"""## Get pretrained model
We can directly use the pretrianed model to do drug repurposing.
"""

import pandas as pd
import numpy as np
import sys
sys.path.insert(1, '../utils')
#from utils import download_and_extract
#download_and_extract()

entity_idmap_file = '/Users/stevensmith/Projects/DRKG/embed/entities.tsv'
relation_idmap_file = '/Users/stevensmith/Projects/DRKG/embed/relations.tsv'

"""## Get embeddings for diseases and drugs"""

# Get drugname/disease name to entity ID mappings
entity_map = {}
entity_id_map = {}
relation_map = {}
with open(entity_idmap_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
    for row_val in reader:
        entity_map[row_val['name']] = int(row_val['id'])
        entity_id_map[int(row_val['id'])] = row_val['name']
        
with open(relation_idmap_file, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
    for row_val in reader:
        relation_map[row_val['name']] = int(row_val['id'])
        
# handle the ID mapping
drug_ids = []
disease_ids = []
for drug in drug_list:
    drug_ids.append(entity_map[drug])
    
for disease in COV_disease_list:
    disease_ids.append(entity_map[disease])

treatment_rid = [relation_map[treat]  for treat in treatment]

# Load embeddings
import torch as th
entity_emb = np.load('/Users/stevensmith/Projects/DRKG/embed/DRKG_TransE_l2_entity.npy')
rel_emb = np.load('/Users/stevensmith/Projects/DRKG/embed/DRKG_TransE_l2_relation.npy')

drug_ids = th.tensor(drug_ids).long()
disease_ids = th.tensor(disease_ids).long()
treatment_rid = th.tensor(treatment_rid)

drug_emb = th.tensor(entity_emb[drug_ids])
treatment_embs = [th.tensor(rel_emb[rid]) for rid in treatment_rid]

"""## Drug Repurposing Based on Edge Score
We use following algorithm to calculate the edge score. Note, here we use logsigmiod to make all scores < 0. The larger the score is, the stronger the $h$ will have $r$ with $t$.

$\mathbf{d} = \gamma - ||\mathbf{h}+\mathbf{r}-\mathbf{t}||_{2}$

$\mathbf{score} = \log\left(\frac{1}{1+\exp(\mathbf{-d})}\right)$

When doing drug repurposing, we only use the treatment related relations.
"""

import torch.nn.functional as fn

gamma=12.0
def transE_l2(head, rel, tail):
    score = head + rel - tail
    return gamma - th.norm(score, p=2, dim=-1)

scores_per_disease = []
dids = []
for rid in range(len(treatment_embs)):
    treatment_emb=treatment_embs[rid]
    for disease_id in disease_ids:
        disease_emb = entity_emb[disease_id]
        score = fn.logsigmoid(transE_l2(drug_emb, treatment_emb, disease_emb))
        scores_per_disease.append(score)
        dids.append(drug_ids)
scores = th.cat(scores_per_disease)
dids = th.cat(dids)

# sort scores in decending order
idx = th.flip(th.argsort(scores), dims=[0])
scores = scores[idx].numpy()
dids = dids[idx].numpy()

"""### Now we output proposed treatments"""

_, unique_indices = np.unique(dids, return_index=True)
topk=100
topk_indices = np.sort(unique_indices)[:topk]
proposed_dids = dids[topk_indices]
proposed_scores = scores[topk_indices]

"""Now we list the pairs of in form of (drug, treat, disease, score) 

We select top K relevent drugs according the edge score
"""

for i in range(topk):
    drug = int(proposed_dids[i])
    score = proposed_scores[i]
    
    print("{}\t{}".format(entity_id_map[drug], score))

"""### Check Clinial Trial Drugs
There are seven clinial trial drugs hit in top100. (Note: Ribavirin exists in DRKG as a treatment for SARS)
"""

# clinical_drugs_file = './COVID19_clinical_trial_drugs.tsv'
# clinical_drug_map = {}
# with open(clinical_drugs_file, newline='', encoding='utf-8') as csvfile:
#     reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['id', 'drug_name','drug_id'])
#     for row_val in reader:
#         clinical_drug_map[row_val['drug_id']] = row_val['drug_name']
        
# for i in range(topk):
#     drug = entity_id_map[int(proposed_dids[i])][10:17]
#     if clinical_drug_map.get(drug, None) is not None:
#         score = proposed_scores[i]
#         print("[{}]\t{}\t{}".format(i, clinical_drug_map[drug],score , proposed_scores[i]))

# len(clinical_drug_map)

