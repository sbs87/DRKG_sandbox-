import csv
import pandas as pd
import numpy as np
import sys


disease_list = [
'Disease::DOID:2841'
]
## Expand to all diseases
drug_list = []
infer_path='biotects_withID' #'../DRKG/drugbank_info/drugbank_biotech.txt'
## curate list
with open(infer_path,newline='',encoding='utf-8') as csvfile:
     reader=csv.DictReader(csvfile,delimiter='\t',fieldnames=['drug','ids'])
     for row_val in reader:
          drug_list.append(row_val['drug'])

treatment = ['Hetionet::CtD::Compound:Disease','GNBR::T::Compound:Disease']

## are these all the combos?

entity_idmap_file = '/Users/stevensmith/Projects/DRKG/embed/entities.tsv'
relation_idmap_file = '/Users/stevensmith/Projects/DRKG/embed/relations.tsv'
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

drug_ids = []
disease_ids = []
for drug in drug_list:
    drug_ids.append(entity_map[drug])
for disease in disease_list:
    disease_ids.append(entity_map[disease])

treatment_rid = [relation_map[treat]  for treat in treatment]

import torch as th
entity_emb = np.load('/Users/stevensmith/Projects/DRKG/embed/DRKG_TransE_l2_entity.npy')
rel_emb = np.load('/Users/stevensmith/Projects/DRKG/embed/DRKG_TransE_l2_relation.npy')


drug_ids = th.tensor(drug_ids).long()
disease_ids = th.tensor(disease_ids).long()
treatment_rid = th.tensor(treatment_rid)

drug_emb = th.tensor(entity_emb[drug_ids])
treatment_embs = [th.tensor(rel_emb[rid]) for rid in treatment_rid]


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
idx = th.flip(th.argsort(scores), dims=[0])
scores = scores[idx].numpy()
dids = dids[idx].numpy()

_, unique_indices = np.unique(dids, return_index=True)

topk=100
topk_indices = np.sort(unique_indices)[:topk]
proposed_dids = dids[topk_indices]
proposed_scores = scores[topk_indices]
for i in range(topk):
    drug = int(proposed_dids[i])
    score = proposed_scores[i]
    
    print("{}\t{}".format(entity_id_map[drug], score))
