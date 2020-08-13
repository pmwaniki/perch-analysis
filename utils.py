import os
from PIL import Image
import pydicom
import warnings
import json
import pandas as pd
from settings import result_dir
import numpy as np
from sklearn.metrics import roc_auc_score
import sqlite3




def read_metrics(file):
    with open(file,'r') as f:
        lines=f.readlines()
    lines2=[json.loads(l) for l in lines]
    dat=pd.DataFrame(lines2)
    return dat

def get_epoch(path):
    metric_file=os.path.join(path,"metrics.txt")
    try:
        metrics=read_metrics(metric_file)
        epoch=metrics['epoch_no'].max()+1
    except Exception as e:
        epoch=0
        warnings.warn("Unable to get epoch. Setting epoch to zero")

    return int(epoch)




def get_model_lr(path,default):
    metric_file = os.path.join(path, "metrics.txt")
    try:
        metrics = read_metrics(metric_file)
        lr = metrics['lr'].iloc[-1]

    except:
        warnings.warn("unable to get learning rate. Setting learning rate to default")
        lr= default

    return float(lr)

# Function for saving results

def create_table3():
    conn=sqlite3.connect("results.db")
    c=conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS table3 (
        method TEXT NOT NULL,
        chestray TEXT NOT NULL,
        model TEXT NOT NULL,
        accuracy NUMERIC,
        auc NUMERIC,
        details TEXT NOT NULL ,
        PRIMARY KEY (method, chestray, model,details)
    )
    """)
    conn.commit()
    conn.close()

def create_table_reviewers():
    conn=sqlite3.connect("results.db")
    c=conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS reviewers (
        method TEXT NOT NULL,
        chestray TEXT NOT NULL,
        model TEXT NOT NULL,
        accuracy TEXT,
        auc TEXT,
        details TEXT NOT NULL ,
        other TEXT,
        PRIMARY KEY (method, chestray, model,details)
    )
    """)
    conn.commit()
    conn.close()


def save_table3(method,chestray,model,accuracy,auc,details=None,other=None):
    create_table3()
    # method="Supervised Pretraining" if supervised else "Unsupervised pretraining"
    with_chestray="With Chestray 14" if chestray else "Without Chestray 14"

    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    if details is not None:
        c.execute("""
                INSERT OR REPLACE INTO table3 (
                    method,
                    chestray,
                    model,
                    accuracy,
                    auc,
                    details,
                    other
                ) VALUES(?,?,?,?,?,?,?)
                """,(method,with_chestray,model,accuracy,auc,details,other))
    else:
        c.execute("""
                        INSERT OR REPLACE INTO table3 (
                            method,
                            chestray,
                            model,
                            accuracy,
                            auc
                        ) VALUES(?,?,?,?,?)
                        """, (method, with_chestray, model, accuracy, auc))
    conn.commit()
    conn.close()

def save_table_reviewers(method,chestray,model,accuracy,auc,details=None):
    create_table_reviewers()
    # method="Supervised Pretraining" if supervised else "Unsupervised pretraining"
    with_chestray="With Chestray 14" if chestray else "Without Chestray 14"

    conn = sqlite3.connect("results.db")
    c = conn.cursor()
    if details is not None:
        c.execute("""
                INSERT OR REPLACE INTO reviewers (
                    method,
                    chestray,
                    model,
                    accuracy,
                    auc,
                    details
                ) VALUES(?,?,?,?,?,?)
                """,(method,with_chestray,model,json.dumps(accuracy),json.dumps(auc),details))
    else:
        c.execute("""
                        INSERT OR REPLACE INTO table3 (
                            method,
                            chestray,
                            model,
                            accuracy,
                            auc
                        ) VALUES(?,?,?,?,?)
                        """, (method, with_chestray, model, accuracy, auc))
    conn.commit()
    conn.close()



