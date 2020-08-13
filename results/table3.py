import sqlite3
import pandas as pd
import numpy as np
from settings import manuscript_dir
import os
import json



conn = sqlite3.connect("results.db")
c=conn.cursor()

c.execute("SELECT * FROM table3" )

rows=c.fetchall()

col_names=list(x[0] for x in c.description)
data=pd.DataFrame(rows,columns=col_names)
data['method']=data.method.map(lambda x:x.replace(" - Ensemble",":Ensemble"))
data=data.loc[data.method.isin(['Multi-task learning', 'Supervised Pretraining',
       'Supervised Pretraining:Ensemble','Multi-task learning:Ensemble',
                                'Unsupervised Pretraining','Unsupervised Pretraining:Ensemble']),:].copy()
data=data.loc[data.method.isin([ 'Supervised Pretraining',
       'Supervised Pretraining:Ensemble',
                                'Unsupervised Pretraining','Unsupervised Pretraining:Ensemble']),:].copy()

data=data.loc[data.details.map(lambda x: (("256" in x) | ("1024" in x))==False),:].copy()

data['Ensemble']=data.method.map(lambda x:"Yes" if "Ensemble" in x else "No")
data['method']=data.method.map(lambda x:x.replace(":Ensemble",""))


data2=data.groupby(['method','chestray','model',"Ensemble"]).apply(lambda d: d[d['accuracy']==max(d['accuracy'])]) #.reset_index()
data3=data2.melt(id_vars=['method','chestray','model',"Ensemble"],value_vars=['auc','accuracy'])
data3['variable']=data3['variable'].map({"auc":"AUC",'accuracy':"Accuracy"})

data3a=data3.loc[~data3.model.isin(['Beta VAE', 'Auto-encoder']),:]
data3a.columns=map(lambda x: x if x != "Ensemble" else "Reader Embeddings",data3a.columns)
data4=pd.pivot_table(data3a,index=['method','model',"Reader Embeddings"],columns=['chestray','variable'],dropna=True)
data4.columns.names=(None,None,None)
data4.columns=data4.columns.droplevel(level=0)
data4b=data4.loc[np.in1d(data4.index.get_level_values(1),['DenseNet121','InceptionV3','ResNet50V2','VAE']),:]
data3b=data3.loc[~data3.model.isin(['Beta VAE', 'Auto-encoder']),:]
data5=pd.pivot_table(data3b,index=['method','model'],columns=['chestray','variable'],dropna=True)
data5.columns.names=(None,None,None)
data5.columns=data5.columns.droplevel(level=0)

with open(os.path.join(manuscript_dir,'table3_ensemble.tex'),'w') as f:
    data4b.to_latex(f,multicolumn=True,index=True,float_format="%.2f",
                   column_format="lll|rr|rr",na_rep='',bold_rows=True,
                   multirow=True)

with open(os.path.join(manuscript_dir,'table3b.tex'),'w') as f:
    data5.to_latex(f,multicolumn=True,index=True,float_format="%.2f",
                   column_format="ll|rr|rr",na_rep='',bold_rows=True,
                   multirow=True)

# c.executemany("INSERT INTO table3 (method,chestray,model,accuracy,auc,details) values(?,?,?,?,?,?)",rows)

ens=data2.copy()
ens=ens.loc[ens['Ensemble']=="Yes",:]
ens=ens.loc[ens['method'] != "Multi-task learning",:]
def get_met(string,metric):
    obj=json.loads(string)
    return obj[metric]


ens['Accuracy:Weighted']=ens['other'].map(lambda x:get_met(x,"weighted")[0])
ens['AUC:Weighted']=ens['other'].map(lambda x:get_met(x,"weighted")[1])
ens['Accuracy:Arbitrators']=ens['other'].map(lambda x:get_met(x,"arbitrators")[0])
ens['AUC:Arbitrators']=ens['other'].map(lambda x:get_met(x,"arbitrators")[1])
ens.rename(mapper={'auc':"AUC:Unweighted","accuracy":"Accuracy:Unweighted"},axis=1,inplace=True)

ens2=ens.melt(id_vars=['method','chestray','model',"Ensemble"],value_vars=['Accuracy:Unweighted', 'AUC:Unweighted',
                                                                           'Accuracy:Weighted', 'AUC:Weighted',
                                                                           'Accuracy:Arbitrators', 'AUC:Arbitrators'])

ens2['Metric']=ens2['variable'].map(lambda x:x.split(":")[0])
ens2['Average']=ens2['variable'].map(lambda x:x.split(":")[1])

ens3=pd.pivot_table(ens2,index=['method','model','Average'],columns=['chestray','Metric'],dropna=True)

ens3.columns.names=(None,None,None)
ens3.columns=ens3.columns.droplevel(level=0)
with open(os.path.join(manuscript_dir,'table4_ensemble.tex'),'w') as f:
    ens3.to_latex(f,multicolumn=True,index=True,float_format="%.2f",
                   column_format="lll|rr|rr",na_rep='',bold_rows=True,
                   multirow=True)

