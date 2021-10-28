#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Requirements
from __future__ import division
from __future__ import print_function
from operator import itemgetter
from itertools import combinations
from IPython.display import display, HTML
import time
import os
from collections import Counter
from collections import defaultdict
from scipy.stats import ks_2samp

import numpy as np
import random
import pandas as pd
import seaborn as sns
import tensorflow as tf
import networkx as nx
import scipy.sparse as sp
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.layers.core import Dropout, Activation
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
#----------------------------------------------------------------------------
# Returns dictionary from combination ID to pair of stitch IDs, 
# dictionary from combination ID to list of polypharmacy side effects, 
# and dictionary from side effects to their names.
def load_combo_se(fname='combo.csv'):
    combo2stitch = {}
    combo2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    print ('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id1, stitch_id2, se, se_name = line.strip().split(',')
        combo = stitch_id1 + '_' + stitch_id2
        combo2stitch[combo] = [stitch_id1, stitch_id2]
        combo2se[combo].add(se)
        se2name[se] = se_name
    fin.close()
    n_interactions = sum([len(v) for v in combo2se.values()])
    print ('Drug combinations: %d Side effects: %d' % (len(combo2stitch), len(se2name)))
    print ('Drug-drug interactions: %d' % (n_interactions))
    return combo2stitch, combo2se, se2name

# Returns dictionary from Stitch ID to list of individual side effects, 
# and dictionary from side effects to their names.
def load_mono_se(fname='mono.csv'):
    stitch2se = defaultdict(set)
    se2name = {}
    fin = open(fname)
    print ('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        contents = line.strip().split(',')
        stitch_id, se, = contents[:2]
        se_name = ','.join(contents[2:])
        stitch2se[stitch_id].add(se)
        se2name[se] = se_name
    return stitch2se, se2name

# Returns dictionary from Stitch ID to list of drug targets
def load_targets(fname='targets-all.csv'):
    stitch2proteins_all = defaultdict(set)
    fin = open(fname)
    print ('Reading: %s' % fname)
    fin.readline()
    for line in fin:
        stitch_id, gene = line.strip().split(',')
        stitch2proteins_all[stitch_id].add(gene)
    return stitch2proteins_all
#----------------------------------------------------------------------------
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')
#----------------------------------------------------------------------------
def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)
#----------------------------------------------------------------------------
#Load data
combo2stitch, combo2se, se2name = load_combo_se()
stitch2se, se2name_mono = load_mono_se()
stitch2proteins_all = load_targets()
#----------------------------------------------------------------------------
#Most common side effects in drug combinations
def get_se_counter(se_map):
    side_effects = []
    for drug in se_map:
        side_effects += list(set(se_map[drug]))
    return Counter(side_effects)
combo_counter = get_se_counter(combo2se)
print("Most common side effects in drug combinations:")
common_se = []
common_se_counts = []
common_se_names = []
for se, count in combo_counter.most_common(964):
    common_se += [se]
    common_se_counts += [count]
    common_se_names += [se2name[se]]
df = pd.DataFrame(data={"Side Effect": common_se, "Frequency in Drug Combos": common_se_counts, "Name": common_se_names})  
display(df)
#---------------------------------------------------------------------------
val_test_size=0.05
n_drugs=645
n_proteins=8934
n_drugdrug_rel_types=964
#---------------------------------------------------------------------------
#list of drugs
lst=[]
for key , value in combo2stitch.items():
  first_name, second_name = map(lambda x: x.strip(), key.split('_'))
  if first_name not in lst:
    lst.append(first_name)
  if second_name not in lst:
    lst.append(second_name)
#---------------------------------------------------------------------------
#list of proteins
p=[]
for k,v in stitch2proteins_all.items():
    for i in v:
        if i not in p:
            p.append(i)
#---------------------------------------------------------------------------
#construct drug-protein-adj matrix
drug_protein_adj=np.zeros((n_drugs,n_proteins))
for i in range(n_drugs):
    for j in stitch2proteins_all[lst[i]]:
        k=p.index(j)
        drug_protein_adj[i,k]=1
#---------------------------------------------------------------------------
#construct drug-drug-adj matrices for all side effects
drug_drug_adj_list=[]
l=[]
for i in range(n_drugdrug_rel_types):
  print(i)
  mat = np.zeros((n_drugs, n_drugs))
  l.append(df.at[i,'Side Effect'])
  for se in l:
    print(se)
    for d1, d2 in combinations(list(range(n_drugs)), 2):
      if lst[d1]+"_"+lst[d2]  in combo2se:
        if se in combo2se[lst[d1]+"_"+lst[d2]]:
            mat[d1,d2]=mat[d2,d1]=1
  l=[]
  drug_drug_adj_list.append(mat)
drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]
#-------------------------------------------------------------------------
#select drug pairs for training & validation & testing
edges=[]
for k in range(n_drugdrug_rel_types):
    l=[]
    for i in range(n_drugs):
        for j in range(n_drugs):
            if drug_drug_adj_list[k][i,j]==1:
                l.append([i,j])
    edges.append(l)
edges_false=[]
for k in range(n_drugdrug_rel_types):
    l=[]
    for i in range(n_drugs):
        for j in range(n_drugs):
            if drug_drug_adj_list[k][i,j]==0:
                l.append([i,j])
    edges_false.append(l)
for k in range(n_drugdrug_rel_types):
    np.random.shuffle(edges[k])
    np.random.shuffle(edges_false[k])
for k in range(n_drugdrug_rel_types):
    a=len(edges[k])
    edges_false[k]=edges_false[k][:a]
edges_all=[]
for k in range(n_drugdrug_rel_types):
    edges_all.append(edges[k]+edges_false[k])
for k in range(n_drugdrug_rel_types):
    np.random.shuffle(edges_all[k])
for k in range(n_drugdrug_rel_types):
    a=len(edges[k])
    edges_all[k]=edges_all[k][:a]
val=[]
test=[]
train=[]
for k in range(n_drugdrug_rel_types):
    a=int(np.floor(len(edges_all[k])*val_test_size))
    val.append(edges_all[k][:a])
    test.append(edges_all[k][a:a+a])
    train.append(edges_all[k][a+a:])
#------------------------------------------------------------------------
#construct drug features
se_mono=[]
for k in se2name_mono:
    se_mono.append(k)
drug_label=np.zeros((n_drugs,len(se_mono)))
for key,value in stitch2se.items():
    j=lst.index(key)
    for v in value:
        i=se_mono.index(v)
        drug_label[j,i]=1
pca = PCA(.95)
pca.fit(drug_label)
pca_= PCA(.95)
pca_.fit(drug_protein_adj)
drug_feat = np.concatenate((pca.transform(drug_label),pca_.transform(drug_protein_adj)),axis=1)
#------------------------------------------------------------------------
#construct train & validation & test sets
val_sets=[]
val_labels=[]
for k in range(n_drugdrug_rel_types):
    v=[]
    a=[]
    for  i in val[k]:
        v.append(drug_feat[i[0]]+drug_feat[i[1]])
        a.append(drug_drug_adj_list[k][i[0],i[1]])
    val_sets.append(v)
    val_labels.append(a)
test_sets=[]
test_labels=[]
for k in range(n_drugdrug_rel_types):
    te=[]
    a=[]
    for  i in test[k]:
        te.append(drug_feat[i[0]]+drug_feat[i[1]])
        a.append(drug_drug_adj_list[k][i[0],i[1]])
    test_sets.append(te)
    test_labels.append(a)
train_sets=[]
train_labels=[]
for k in range(n_drugdrug_rel_types):
    tr=[]
    a=[]
    for  i in train[k]:
        tr.append(drug_feat[i[0]]+drug_feat[i[1]])
        a.append(drug_drug_adj_list[k][i[0],i[1]])
    train_sets.append(tr)
    train_labels.append(a)
val_org=[]
val_label_org=[]
test_org=[]
test_label_org=[]
train_org=[]
train_label_org=[]
for k in range(n_drugdrug_rel_types):
    val_org.append(np.array(val_sets[k]))
    val_label_org.append(np.array(val_labels[k]))
    test_org.append(np.array(test_sets[k]))
    test_label_org.append(np.array(test_labels[k]))
    train_org.append(np.array(train_sets[k]))
    train_label_org.append(np.array(train_labels[k]))
#-----------------------------------------------------------------------
#construct model
model = Sequential()
model.add(Dense(input_dim=drug_feat.shape[1], kernel_initializer='glorot_normal',units=300))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(input_dim=300, kernel_initializer='glorot_normal', units=200))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(input_dim=200, kernel_initializer='glorot_normal', units=100))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(input_dim=100, kernel_initializer='glorot_normal', units=1))
model.add(Activation('sigmoid'))
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
#------------------------------------------------------------------------
#get criteria
roc_score=[]
aupr_score=[]
f_score=[]
thr=[]
precision=[]
recall=[]
tpos=[]
fpos=[]
tneg=[]
fneg=[]
acc=[]
mcc=[]
for k in range(n_drugdrug_rel_types):
    print(k)
    model.fit(train_org[k],train_label_org[k],batch_size=1024, epochs=5)
    model.evaluate(val_org[k], val_label_org[k])
    model.evaluate(test_org[k], test_label_org[k])
    roc=metrics.roc_auc_score(test_label_org[k],model.predict(test_org[k]))
    roc_score.append(roc)
    aupr=metrics.average_precision_score(test_label_org[k],model.predict(test_org[k]))
    aupr_score.append(aupr)
    fpr, tpr, thresholds=metrics.roc_curve(test_label_org[k],model.predict(test_org[k]))
    scores=[metrics.f1_score(test_label_org[k], to_labels(model.predict(test_org[k]), t)) for t in thresholds]
    ma= max(scores)
    f_score.append(ma)
    idx=np.argmax(scores)
    bt=thresholds[idx]
    thr.append(bt)
    p=metrics.precision_score(test_label_org[k], to_labels(model.predict(test_org[k]), bt))
    precision.append(p)
    r=metrics.recall_score(test_label_org[k], to_labels(model.predict(test_org[k]), bt))
    recall.append(r)
    TP, FP, TN, FN=perf_measure(test_label_org[k],to_labels(model.predict(test_org[k]), bt))
    tpos.append(TP)
    fpos.append(FP)
    tneg.append(TN)
    fneg.append(FN)
    ac = float(TP + TN)/(TP+FP+FN+TN)
    acc.append(ac)
    mc=metrics.matthews_corrcoef(test_label_org[k],to_labels(model.predict(test_org[k]), bt))
    mcc.append(mc)
model.save('model.h5')
loaded_model =load_model('model.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
for k in range(541):
    print(k)
    model.fit(train_org[k],train_label_org[k],batch_size=1024, epochs=5)
    model.evaluate(val_org[k], val_label_org[k])
    model.evaluate(test_org[k], test_label_org[k])
    roc=metrics.roc_auc_score(test_label_org[k],model.predict(test_org[k]))
    roc_score[k]=roc
    aupr=metrics.average_precision_score(test_label_org[k],model.predict(test_org[k]))
    aupr_score[k]=aupr
    fpr, tpr, thresholds=metrics.roc_curve(test_label_org[k],model.predict(test_org[k]))
    scores=[metrics.f1_score(test_label_org[k], to_labels(model.predict(test_org[k]), t)) for t in thresholds]
    ma= max(scores)
    f_score[k]=ma
    idx=np.argmax(scores)
    bt=thresholds[idx]
    thr[k]=bt
    p=metrics.precision_score(test_label_org[k], to_labels(model.predict(test_org[k]), bt))
    precision[k]=p
    r=metrics.recall_score(test_label_org[k], to_labels(model.predict(test_org[k]), bt))
    recall[k]=r
    TP, FP, TN, FN=perf_measure(test_label_org[k],to_labels(model.predict(test_org[k]), bt))
    tpos[k]=TP
    fpos[k]=FP
    tneg[k]=TN
    fneg[k]=FN
    ac = float(TP + TN)/(TP+FP+FN+TN)
    acc[k]=ac
    mc=metrics.matthews_corrcoef(test_label_org[k],to_labels(model.predict(test_org[k]), bt))
    mcc[k]=mc

