"""
This script will compute TransE scores for a given node-edge-node triplet from DRKG. 
The nodes and edges are input from lists within seperate files. The nodes and edges are definef in DRKG. 

Author Steven Smith, PhD

"""

import csv
import pandas as pd
import numpy as np
import sys
import csv
import torch as th
import torch.nn.functional as fn


# Read lists, mapping name -> unique ID
def read_elements(element_path,headers,keyheader):
    element_list=[]
    with open(element_path) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t',fieldnames=headers)
        for row in reader:
            element_list.append(row[keyheader])
    return(element_list)

# Computes transE_l2 score: gamma - ||(head + edge - tail)||
def transE_l2(head, edge, tail, gamma):
    score = head + edge - tail
    return gamma - th.norm(score, p=2, dim=-1)

# Map node or edge names to uniuq IDs, handling unknown edge or node names gracefully. 
def map_components(component_list, mapping_file):
    ids=[]
    for component in component_list:
        try:
            ids.append(mapping_file[component])
        except:
            print("WARNING: skipping unknown "+component)
    return(ids)

# Set global variables, structures

gamma=12.0
entity_mapping_path = '/Users/stevensmith/Projects/DRKG/embed/entities.tsv'
edge_mapping_path = '/Users/stevensmith/Projects/DRKG/embed/relations.tsv'
entity_embeddings = '/Users/stevensmith/Projects/DRKG/embed/DRKG_TransE_l2_entity.npy'
edge_embeddings = '/Users/stevensmith/Projects/DRKG/embed/DRKG_TransE_l2_relation.npy'

entity_map = {}
entity_id_map = {}
edge_map = {}
edge_id_map = {}


# Read node and edge list paths. 

node1_path=sys.argv[1]
edge_path=sys.argv[2]
node2_path=sys.argv[3]


# Rare node1, Mantle cell lymphoma. Associated with translocation in CCND1. Treated with chemotherapy and antibodies 
# https://www.orpha.net/consor/cgi-bin/node1_Search.php?lng=EN&data_id=10693&MISSING%20CONTENT=Mantle-cell-lymphoma&search=node1_Search_Simple&title=Mantle%20cell%20lymphoma
#3 TOP RESULS DB09107 - Methoxy polyethylene glycol-epoetin beta, stimulates blood cells

# Read node and edge lists. 

node1_list=read_elements(node1_path,['node1'],'node1')
edge_list=read_elements(edge_path,['edge'],'edge')
node2_list=read_elements(node2_path,['node2','ids'],'node2')


# Read in node, edge mapping files

with open(entity_mapping_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
    for row_val in reader:
        entity_map[row_val['name']] = int(row_val['id'])
        entity_id_map[int(row_val['id'])] = row_val['name']

with open(edge_mapping_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', fieldnames=['name','id'])
    for row_val in reader:
        edge_map[row_val['name']] = int(row_val['id'])
        edge_id_map[int(row_val['id'])] = row_val['name']


# Map nodes and edges to their unique ids, convert to tensors

node1_ids = th.tensor(map_components(node1_list,entity_map)).long()
node2_ids = th.tensor(map_components(node2_list,entity_map)).long()
edge_ids = map_components(edge_list,edge_map)


#node2_ids = th.tensor(node2_ids).long()
node1_ids = th.tensor(node1_ids).long()
edge_ids = th.tensor(edge_ids).long()

# Read in entity and edge embeddings
entity_emb = np.load( entity_embeddings)
edge_emb = np.load(edge_embeddings)

# Filter for node1, node2 and edge embeddings of interest
node1_embs = {rid:th.tensor(entity_emb[rid]) for rid in node1_ids}
node2_embs = {rid:th.tensor(entity_emb[rid]) for rid in node2_ids}
edge_embs = {rid:th.tensor(edge_emb[rid]) for rid in edge_ids}



""""
Compute TransE
"""

# Compute node1-edge-node2 transE score and print. 
for (node1_i, node1_i_emb) in node1_embs.items():
    for (edge_i, edge_i_emb) in edge_embs.items():
        for(node2_i, node2_i_emb) in node2_embs.items():
            score = fn.logsigmoid(transE_l2(node1_i_emb, edge_i_emb, node2_i_emb, gamma))
            node1_name=entity_id_map[int(node1_i)]
            edge_name=edge_id_map[int(edge_i)]
            node2_name=entity_id_map[int(node2_i)]
            print("{}\t{}\t{}\t{}".format(node1_name,edge_name,node2_name,score))

## END
