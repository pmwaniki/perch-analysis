import pandas as pd
import numpy as np
from settings import perch_config,rsna_config,chestray_config,label_var,path_var,label_sep
import json
import os


class Dataset:
    def __init__(self,train_csv,test_csv=None,multilabel=False):
        self.train_csv=train_csv
        self.test_csv=test_csv
        self.train=pd.read_csv(train_csv)
        self.multilabel=multilabel
        if test_csv is not None:
            self.test=pd.read_csv(test_csv)

    def load_data(self,labels=True,include_test=True):
        path_train=self.train[path_var].values
        path_test = None if self.train_csv is None else self.test[path_var].values
        if labels:
            labs_train=self.train[label_var]
            labs_test=None if self.test_csv is None else self.test[label_var]
            if (self.multilabel):
                labs_train= np.array(labs_train)
                labs_train=list(map(lambda x:x.split("|"),labs_train))
                labs_train=np.array(labs_train,dtype="float")

                labs_test= None if self.train_csv is None else np.array(labs_test)
                labs_test = list(map(lambda x: x.split("|"), labs_test))
                labs_test = np.array(labs_test, dtype="float")
            else:
                labs_train= pd.get_dummies(labs_train).values
                labs_test = None if self.train_csv is None else pd.get_dummies(labs_test).values

        if include_test:
            if labels:
                return path_train,labs_train,path_test,labs_test
            else:
                return path_train,path_test
        else:
            if labels:
                return path_train,labs_train
            else:
                return path_train

    # def load_weights(self,include_test=True):
    #     with open(os.path.join(perch_config.image_path, perch_config.class_weights),'r') as f:
    #         weights=json.load(f)
    #     return weights



# self=Dataset(rsna_config.train_csv,rsna_config.validation_csv)
# self=Dataset(perch_config.train_csv,perch_config.validation_csv)
# self=Dataset(train_csv=chestray_config.train_csv,test_csv= chestray_config.validation_csv,multilabel=True)

#rsna=Dataset(rsna_config.train_csv,rsna_config.validation_csv)
perch=Dataset(perch_config.train_csv,perch_config.validation_csv)
chestray=Dataset(train_csv=chestray_config.train_csv,test_csv= chestray_config.validation_csv,multilabel=True)

perch.train_long=pd.read_csv(os.path.join(perch_config.image_path, "Assessors_train.csv"))
perch.test_long=pd.read_csv(os.path.join(perch_config.image_path, "Assessors_test.csv"))