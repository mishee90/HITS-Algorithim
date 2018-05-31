# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:01:43 2018

@author: dhavy
"""

import numpy as np
import scipy.sparse as sparse
import time
import pickle
#import igraph
#from dataset_fetcher import ListToMatrixConverter
import matplotlib.pyplot as plt
import matplotlib.patches as mp
import time
import io


is_sparse = True
epsilon = 1e-10
show_iters = False

datapath = 'C:/Dhaval/Program Course Work/Linear Algebra/Project/HITS-Algorithim/HITS-Algorithm-implementation'

users_path = datapath+'/data/users'
map_path = datapath+'/data/map'
sparse_link_matrix_path = datapath+'/data/sparse_link_matrix'
dense_link_matrix_path = datapath+'/data/dense_link_matrix'

if is_sparse:
    link_matrix_path = sparse_link_matrix_path
else:
    link_matrix_path = dense_link_matrix_path


with io.open(users_path,'rb')as f:
    users = pickle.load(f)

with io.open(map_path,'rb') as f:
    index_id_map = pickle.load(f)

with io.open(link_matrix_path,'rb') as f:
    if is_sparse==True:
        link_matrix = sparse.load_npz(link_matrix_path)
    else:
        link_matrix = np.load(link_matrix_path)



### Calculating the scores
epsilon = 0.0001
n = link_matrix.shape[0]
link_matrix_tr = link_matrix.transpose()

epsilon_matrix = epsilon * np.ones(n)
hubs = np.ones(n)
auths = np.ones(n)
size = 30
all_hubs = [0]
all_auths = [0]

# Below code calculates the hubbiness and auths

if is_sparse:
    while True:
        hubs_old = hubs
        auths_old = auths
        auths = link_matrix_tr * hubs_old
        max_score = auths.max(axis=0)
        if max_score != 0:
            auths = auths / max_score
            all_auths.append(auths)
            hubs = link_matrix * auths
            max_score = hubs.max(axis=0)
            if max_score != 0:
                hubs = hubs / max_score
                all_hubs.append(hubs)
                
            if (((abs(hubs - hubs_old)) < epsilon_matrix).all()) and (((abs(auths - auths_old)) < epsilon_matrix).all()):
                break



