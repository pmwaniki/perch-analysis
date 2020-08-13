import numpy as np
from sklearn.metrics import roc_auc_score,f1_score,recall_score,precision_score,accuracy_score,cohen_kappa_score,confusion_matrix
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt

def multilabel_auc(ytrue, ypred):
    auc = []
    for i in range(ytrue.shape[1]):
        try:
            auc.append(roc_auc_score(ytrue[:, i], ypred[:, i]))
        except Exception as e:
            auc.append(np.nan)

    return auc


def multilabel_acc(ytrue, ypred):
    acc = []
    for i in range(ytrue.shape[1]):
        acc.append(np.mean(ytrue[:, i]== (ypred[:, i]>=0.5)))
    return acc

def consolidation_infiltrates_acc(ytrue, ypred):
    y_true_cons=ytrue[:,0]+ytrue[:,2]
    y_true_infil = ytrue[:, 1] + ytrue[:, 2]
    y_pred_cons = ypred[:, 0] + ypred[:, 2]
    y_pred_infil = ypred[:, 1] + ypred[:, 2]
    uninterpretable=np.apply_along_axis(lambda x:x[4]==1,axis=1,arr=ytrue)
    accuracy_consolidation=np.mean(y_true_cons[~uninterpretable]==(y_pred_cons[~uninterpretable]>=0.5))
    accuracy_infil = np.mean(y_true_infil[~uninterpretable] == (y_pred_infil[~uninterpretable] >= 0.5))
    kappa_consolidation = cohen_kappa_score(y_true_cons[~uninterpretable] ,(y_pred_cons[~uninterpretable] >= 0.5)*1)
    kappa_infil = cohen_kappa_score(y_true_infil[~uninterpretable] , (y_pred_infil[~uninterpretable] >= 0.5)*1)
    return {"acc_kappa_consolidation":[accuracy_consolidation,kappa_consolidation],'acc_kappa_infiltrates':[accuracy_infil,kappa_infil]}


def multiclass_acc( ytrue, ypred):
    ypred=np.argmax(ypred,axis=1)
    ytrue=np.argmax(ytrue,axis=1)
    return np.mean(ytrue == ypred )


def multilabel_precision(ytrue,ypred):
    ypred=(ypred>0.5)*1.
    with warnings.catch_warnings(record=True) as w:
        result=precision_score(ytrue,ypred,average=None)
        len(w)
    return result


def multilabel_recall(ytrue,ypred):
    ypred=(ypred>0.5)*1.
    return recall_score(ytrue,ypred,average=None)


def multilabel_f1(ytrue,ypred):
    ypred=(ypred>0.5)*1.
    return f1_score(ytrue,ypred,average=None)


def categorical_loss(ytrue,ypred):
    l=tf.keras.losses.categorical_crossentropy(ytrue,ypred)
    return l.numpy().mean()

def binary_loss(ytrue,ypred):
    l=tf.keras.losses.binary_crossentropy(ytrue,ypred)
    return l.numpy().mean()

def binary_acc(ytrue,y_pred):
    a=tf.metrics.binary_accuracy(ytrue,y_pred)
    return a.numpy().mean()

def categorical_acc(ytrue,ypred):
    a=tf.metrics.categorical_accuracy(ytrue,ypred)
    return a.numpy().mean()


def perch_confusion_matrix(ytrue,ypred,labels=['Consolidation',
                                                'Other Infiltrate',
                                                'Consolidation and \nOther Infiltrate',
                                                'Normal',
                                                'Uninterpretable']):
    cf_matrix=confusion_matrix(ytrue,ypred,labels=[0,1,2,3,4])
    cf_matrix_norm=cf_matrix/np.reshape(cf_matrix.sum(axis=1),[-1,1])

    fig,ax=plt.subplots(1,1,figsize=(8,7))
    img=ax.matshow(cf_matrix_norm,cmap=plt.cm.get_cmap("Greys"),vmin=0,vmax=1)
    for i in [0,1,2,3,4]:
        for j in [0,1,2,3,4]:
            ax.text(j,i,"%d(%.1f)" % (cf_matrix[i,j],cf_matrix_norm[i,j]),ha='center',fontsize=9)
    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(labels,rotation=45,ha="left",rotation_mode="anchor")
    # ax.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #      rotation_mode="anchor")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(labels)
    ax.set_ylim(4.5,-0.5)
    ax.set_xlabel("Predicted Class",fontsize=14)
    ax.set_ylabel("Target Class",fontsize=14)
    fig.colorbar(img)
    plt.show()

    return fig

