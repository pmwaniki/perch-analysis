#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 12:29:45 2018

@author: pmwaniki


Prepare data:
    
    create train test split
"""

import numpy as np
import pandas as pd
import os
import pickle
from settings import perch_config, chestray_config, path_var, dropbox_dir, label_var, label_sep

from sklearn.utils import class_weight
import json


def get_class_weights(labels, classes, multi_label=False, normalize=True):
    if not multi_label:
        unique = labels.value_counts(normalize=True)
        weights = unique ** (-1)
        if normalize:
            weights = weights / weights.min()
        return {i: weights[i] for i in classes}
    else:
        result=[]
        target2 = [i.split(label_sep) for i in labels]
        target3 = np.array(target2).astype("float")
        weights=np.apply_along_axis(lambda x: class_weight.compute_class_weight("balanced",np.array([0,1]),x),0,target3)
        for i in range(weights.shape[1]):
            result.append({0:weights[0,i],1:weights[1,i]})

        return result




# CHESTXRAY8
chestray = pd.read_csv(os.path.join(chestray_config.image_path, 'Data_Entry_2017.csv'))

train_list=pd.read_csv(os.path.join(chestray_config.image_path, 'train_val_list.txt'),sep='\t',header=None).iloc[:,0].values
test_list=pd.read_csv(os.path.join(chestray_config.image_path, 'test_list.txt'),sep='\t',header=None).iloc[:,0].values


chestray[path_var] = list(map(lambda x: os.path.join(chestray_config.image_path, "images/", x), chestray['Image Index']))
chestray[label_var]=chestray['Finding Labels'].map(lambda x:x.split("|"))
chestray[label_var]=chestray[label_var].map(lambda x: ["1" if value in x else "0" for key,value in chestray_config.labels.items()])
chestray[label_var]=chestray[label_var].map(lambda x:label_sep.join(x))

chestray_train=chestray.loc[chestray['Image Index'].isin(train_list),:]
chestray_test = chestray.loc[chestray['Image Index'].isin(test_list),:]

chestray_train.to_csv(os.path.join(chestray_config.image_path, chestray_config.train_csv), index=False)
chestray_test.to_csv(os.path.join(chestray_config.image_path, chestray_config.validation_csv), index=False)



# PERCH
assessors=pd.read_excel(os.path.join(perch_config.image_path, "PERCHCXR_RevIDs.xlsx"))
bangladesh = pd.read_csv(os.path.join(perch_config.image_path, "./Bangladesh/0_Data/PERCH_CXR_BAN.CSV"))
south_africa = pd.read_csv(os.path.join(perch_config.image_path, "./South Africa/0_Data/PERCH_CXR_SAF.CSV"))
mali = pd.read_csv(os.path.join(perch_config.image_path, "./Mali/0_Data/PERCH_CXR_MAL.CSV"))
zambia = pd.read_csv(os.path.join(perch_config.image_path, "./Zambia/0_Data/PERCH_CXR_ZAM.CSV"))
kenya = pd.read_csv(os.path.join(perch_config.image_path, "./Kenya/0_Data/PERCH_CXR_KEN.CSV"))
thailand = pd.read_csv(os.path.join(perch_config.image_path, "./Thailand/0_Data/PERCH_CXR_THA.CSV"))
gambia = pd.read_csv(os.path.join(perch_config.image_path, "./Gambia/0_Data/PERCH_CXR_GAM_CRES.CSV"))

image_files = {}
for root, dirs, files in os.walk(perch_config.image_path):
    for file in files:
        if file.lower().endswith("jpg"):
            image_files[file.split(".")[0]] = file

perch = pd.concat([bangladesh, south_africa, mali, zambia, kenya, thailand,gambia])
perch = perch[~((perch.DATANOIMG == 1) | (perch.IMGNODATA == 1))]
perch['directory'] = perch.SITE.map(
    {'BAN': "Bangladesh", 'SAF': "South Africa", 'MAL': "Mali",
     'ZAM': 'Zambia', 'KEN': 'Kenya', 'THA': 'Thailand','GAM':'Gambia'})
perch['file_name'] = perch.CXRIMGID.map(image_files)
perch[path_var] = perch[['directory', 'file_name']].aggregate(
    lambda x: np.nan if pd.isna(x[1]) else os.path.join(perch_config.image_path,x[0], x[1]), axis=1)
perch[label_var] = perch.FINALCONC.astype("int")-1

assessors2=pd.merge(assessors,perch[['CXRIMGID','PATID','REV1', 'REV2','ARB1', 'ARB2','labels','path']],
                    how='right',on="CXRIMGID",suffixes=("_id","_value"))
assessors3a=pd.melt(assessors2,id_vars=['CXRIMGID','PATID','labels','path'],
                   value_vars=['REV1_id', 'REV2_id', 'ARB1_id', 'ARB2_id'],
                   var_name='rev_variable',value_name='rev_id')
assessors3a.dropna(axis=0,how='all',subset=['rev_id'],inplace=True)
assessors3a['rev_variable']=assessors3a['rev_variable'].replace('_id$','',regex=True)
assessors3a.rename(columns={'rev_variable':'reviewer'},inplace=True)


assessors3b=pd.melt(assessors2,id_vars=['CXRIMGID','PATID','labels','path'],
                   value_vars=['REV1_value', 'REV2_value', 'ARB1_value', 'ARB2_value'],
                   var_name='value_var',value_name='rev_value')
assessors3b.dropna(axis=0,how='all',subset=['rev_value'],inplace=True)
assessors3b['value_var']=assessors3b['value_var'].replace('_value$','',regex=True)
assessors3b.rename(columns={'value_var':'reviewer'},inplace=True)

assessors3=pd.merge(assessors3a,assessors3b,on=['CXRIMGID','PATID','labels','path','reviewer'])
assessors3['rev_label']=assessors3['rev_value'].map(lambda x: int(x-1))
assessors3['reviewer']=assessors3['rev_id'].map({v:k for k,v in enumerate(np.sort(assessors3['rev_id'].unique()))})

accuracy=assessors3.groupby('rev_id').apply(lambda df: np.mean(df['labels']==df['rev_label']))
accuracy.agg({'min':np.min,'max':np.max,'median':np.median})


assessor_summary=assessors3.groupby('rev_id').apply(lambda df: pd.Series({
    'prop':np.mean(df['labels']==df['rev_label']),
    'N':df.shape[0]}))
#summary of number of images assesses
assessor_summary['N'][:14].agg(['min','max'])
assessor_summary['N'][14:].agg(['min','max'])
#summry of accuracy
assessor_summary['prop'][:14].agg(['median','min','max'])
assessor_summary['prop'][14:].agg(['median','min','max'])
#ARBITATOR AGREEMENT
perch.dropna(subset=['REV1','REV2']).apply(lambda df2:df2['REV1']==df2['REV2'],axis=1).mean()
perch.dropna(subset=['ARB1','ARB2']).apply(lambda df2:df2['ARB1']==df2['ARB2'],axis=1).mean()

# 3 train test split
# split by patient id
np.random.seed(123)
patient_ids = perch.PATID.unique()
test_ids = np.random.choice(patient_ids, 500, replace=False)

perch_test = perch[perch.PATID.isin(test_ids)].copy()
perch_train = perch[~perch.PATID.isin(test_ids)].copy()

assessors3_test = assessors3[assessors3.PATID.isin(test_ids)].copy()
assessors3_train = assessors3[~assessors3.PATID.isin(test_ids)].copy()

perch_weights=class_weight.compute_class_weight("balanced",np.array([0,1,2,3,4]),perch_train[label_var])
perch_weights={i:perch_weights[i] for i in range(5)}
perch_train.loc[:,'sample_weight']=class_weight.compute_sample_weight("balanced",perch_train[label_var])
perch_test.loc[:,'sample_weight']=class_weight.compute_sample_weight("balanced",perch_test[label_var])


perch_train.to_csv(os.path.join(perch_config.image_path, perch_config.train_csv), index=False)
perch_test.to_csv(os.path.join(perch_config.image_path, perch_config.validation_csv), index=False)

assessors3_train.to_csv(os.path.join(perch_config.image_path, "Assessors_train.csv"), index=False)
assessors3_test.to_csv(os.path.join(perch_config.image_path, "Assessors_test.csv"), index=False)


with open(os.path.join(perch_config.image_path, perch_config.class_weights), 'w') as f:
    json.dump(perch_weights, f)

with open(os.path.join(dropbox_dir,"embeddings/data/labels.pkl"),'wb') as f:
    pickle.dump([perch_train[label_var].values,perch_test[label_var].values],f)