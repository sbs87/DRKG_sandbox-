import numpy as np
import pandas as pd
import dgl
import sys
sys.path.insert(1, '/Users/stevensmith/Projects/DRKG/utils')
from utils import download_and_extract
entity_emb = np.load('/Users/stevensmith/Projects/DRKG/embed/DRKG_TransE_l2_entity.npy')
rel_emb = np.load('/Users/stevensmith/Projects/DRKG/embed/DRKG_TransE_l2_relation.npy')

drkg_file = '/Users/stevensmith/Projects/DRKG/drkg.tsv'
df = pd.read_csv(drkg_file, sep="\t")
triplets = df.values.tolist()