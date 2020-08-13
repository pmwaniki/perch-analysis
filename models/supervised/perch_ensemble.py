import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
from models.supervised.applications import PerchEnsembleModel
from models.supervised.applications import vgg16,resnet50,resnet101,densnet121,inceptionv3,resnet50v2
from settings import perch_config,result_dir
from utils import save_table3,save_table_reviewers
from callbacks import CyclicLR
import os
from sklearn.metrics import classification_report
from metrics import multilabel_auc,multiclass_acc,consolidation_infiltrates_acc,perch_confusion_matrix
import matplotlib.pyplot as plt
import json
import pickle
from data.datasets import perch
# import tensorflow as tf
from data.preprocess_image import load_and_preprocess_image_ensemble,flip,rotate,color,crop_and_pad,augment_ensemble
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
from functools import partial
from scipy.cluster.hierarchy import dendrogram,linkage
import seaborn as sns
display=os.environ.get("DISPLAY",None)
AUTOTUNE=tf.data.experimental.AUTOTUNE



basemodel=inceptionv3
weights_= 'imagenet'
init_weights='imagenet'
label_smoothening=True
conditioning_="multiply"  # add multiply
activation_conditioning_= 'sigmoid' # None, sigmoid,relu

warm_start=False
pooling_='avg'
l2_=0.0005  #0.05, 0.0005, 0.00005
l2_embed=0.0001
crop_prop=0.5 # 0.2,0.5, 0.9
z=16
initial_lr=0.1
M=6;size=50
if weights_ not in [None, 'imagenet', 'chestnet'] : raise Exception("Unrecognised weight initialization")
with_chestray=True if weights_=='chestnet' else False

details_="activation: {} | l2: {:.5f} | l2_emb: {:.6f} | crop: {:.2f} | weights: {} z_dim: {} |lr {:.2f}".format(activation_conditioning_,l2_,l2_embed,crop_prop,weights_,z,initial_lr)
weight_path="clr_ensemble_{}_l2 {:.5f}_crop {:.2f} _ weights {}_lr {:.2f}_activation {}.pkl".format(basemodel.name,
                                                                                             l2_,crop_prop,weights_,
                                                                                               initial_lr,activation_conditioning_)
if label_smoothening:
    weight_path=weight_path.replace(".pkl","_smooth.pkl")
    details_=details_ + "_smooth"

input_shape=basemodel.input_shape

model=PerchEnsembleModel(base_model=basemodel,n_out=5,n_embedding=z,
                         weights=weights_,init_weights=init_weights,pooling=pooling_,l2=l2_,
                         l2_embeddings=l2_embed,conditioning=conditioning_,
                         activation_conditioning=activation_conditioning_)
model.build([(None,)+input_shape,(None,1)])

perch_train=perch.train_long
perch_test=perch.test_long

path_train=perch_train['path']
labs_train=pd.get_dummies(perch_train['rev_label']).values
rev_train=perch_train['reviewer']
correct_train=(perch_train['labels']==perch_train['rev_label']).values*1.0

path_test=perch_test['path']
labs_test=pd.get_dummies(perch_test['rev_label']).values
rev_test=perch_test['reviewer']
correct_test=(perch_test['labels']==perch_test['rev_label']).values*1.0






BATCH_SIZE=24

image_train=tf.data.Dataset.from_tensor_slices((path_train,rev_train))
image_test=tf.data.Dataset.from_tensor_slices((path_test,rev_test))

label_train=tf.data.Dataset.from_tensor_slices(tf.cast(labs_train,tf.float32))
label_test=tf.data.Dataset.from_tensor_slices(tf.cast(labs_test,tf.float32))



load_and_preprocess_image_ensemble_=partial(load_and_preprocess_image_ensemble,
                                            shape=input_shape)

image_train_ds=image_train.map(load_and_preprocess_image_ensemble_,
                               num_parallel_calls=AUTOTUNE)
image_test_ds=image_test.map(load_and_preprocess_image_ensemble_,
                             num_parallel_calls=AUTOTUNE)

augumentations=[
    partial(color,cont_lower=0.3,cont_upper=0.99,bright_delta=0.01),
    # partial(crop_and_pad,proportion=0.95,width=.5,height=.5),
    partial(crop_and_pad,proportion=crop_prop,width=.5,height=.5),
    # partial(crop_and_resize,width=0.7,height=0.7),
    flip,
    rotate,
    partial(tf.subtract,y=0.5),
    partial(tf.multiply,y=2.0),


]

augumentations_test=[
    partial(tf.subtract,y=0.5),
    partial(tf.multiply,y=2.0),
]



augment_ensemble_=partial(augment_ensemble,aug_funs=augumentations)
augment_test_ensemble_=partial(augment_ensemble,aug_funs=augumentations_test)


image_train_ds=image_train_ds.map(augment_ensemble_)
image_test_ds=image_test_ds.map(augment_test_ensemble_)

ss=[im for im,rev in image_train_ds.batch(25).take(1)][0]
ss2=np.zeros((input_shape[0]*5,input_shape[1]*5,3))
k=0
for i in range(5):
    for j in range(5):
        ss2[i*input_shape[0]:(i+1)*input_shape[0],j*input_shape[1]:(j+1)*input_shape[1]]=ss[k,:,:]
        k=k+1
if display:
    plt.imshow(ss2/2.0+0.5)
    plt.show()



ds_train = tf.data.Dataset.zip((image_train_ds,label_train))
ds_test= tf.data.Dataset.zip((image_test_ds,label_test ))





ds_train=ds_train.shuffle(buffer_size=4500)
ds_train=ds_train.repeat()
ds_train=ds_train.batch(BATCH_SIZE)
ds_train=ds_train.prefetch(buffer_size=AUTOTUNE)

ds_test=ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)



optimizer=tf.keras.optimizers.SGD(lr=initial_lr)




if label_smoothening:
    model.compile(optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=["accuracy"])
else:
    model.compile(optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])

swa_callback=CyclicLR(a0=initial_lr,M=M,size=size)
print("Minimum learning rate: %7.6f" % np.min(swa_callback.lr_list))

if warm_start:
    with open(os.path.join(result_dir,"finetune",weight_path),"rb") as f:
        saved_weights=pickle.load(f)
    swa_callback.weight_list=saved_weights
    model.set_weights(saved_weights[-1])



model.fit(ds_train,
          epochs=M*size,
          steps_per_epoch=int(len(path_train)/BATCH_SIZE),
          validation_data=ds_test,validation_steps=int(len(path_test)/BATCH_SIZE),
          # class_weight=class_weights,
          callbacks=[swa_callback],
          verbose=1
          )
with open(os.path.join(result_dir,"finetune",weight_path),"wb") as f:
    pickle.dump(swa_callback.weight_list,f)

fig,axs=plt.subplots(nrows=1,ncols=2)
axs[0].plot(model.history.history['loss'])
axs[0].plot(model.history.history['val_loss'])
axs[0].set_title("loss")
axs[0].set_ylim(top=min(3,np.max(model.history.history['val_loss'])),bottom=np.min(model.history.history['loss']))
axs[1].plot(model.history.history['accuracy'])
axs[1].plot(model.history.history['val_accuracy'])
axs[1].set_title("accuracy")
if display:
    plt.show()
else:
    plt.savefig("/home/pmwaniki/Dropbox/tmp/perch_ensemble_%s_%s.png" % (basemodel.name,details_))





#predict ensemble

pred_all=[]
for i in range(18):
    print("Making prediction for reviewer %d ..." % (i+1))
    ds_test_=tf.data.Dataset.from_tensor_slices((perch.test['path'],[i]*perch.test.shape[0])).\
        map(load_and_preprocess_image_ensemble_).map(augment_test_ensemble_).batch(100).prefetch(AUTOTUNE)
    pred_ens = []
    for w in swa_callback.weight_list:
        model.set_weights(w)
        p=[model.predict(k) for k in ds_test_]
        pred_ens.append(np.concatenate(p,axis=0))
    pred_ens2=np.stack(pred_ens)
    pred_all.append(pred_ens2.mean(axis=0))



# embeddings
embeddings=[i for i in model.weights if 'embedding' in i.name][0].numpy()
plt.matshow(embeddings)
plt.colorbar()
plt.show()
#
# #dendogram
linked = linkage(embeddings, 'complete',metric='cosine')

labelList = range(1, 19)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
plt.show()

#reviewer accuracy
rev_acc=perch_train.groupby('reviewer').apply(lambda df: np.mean(df['labels']==df['rev_label'])).to_dict()
rev_weights={k:v/np.sum(list(rev_acc.values())) for k,v in rev_acc.items() }
rev_weights=[rev_weights[i] for i in range(18)]



#accuracy for each rater
rev_acc=[multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_all[i]) for i in range(18)]
rev_auc=[np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_all[i])) for i in range(18)]
# aggregate prediction
pred_stacked = np.stack(pred_all)




pred_ensemble_arbitrators=pred_stacked[14:,:,:].mean(axis=0)
pred_ensemble=pred_stacked.mean(axis=0)
pred_ensemble_weighted=np.average(pred_stacked,axis=0,weights=np.array(rev_weights))
accuracy=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble)
accuracy_consolidation_infiltrates=consolidation_infiltrates_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble)
accuracy_arbitrators=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_arbitrators)
accuracy_weighted=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_weighted)
auc=np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_ensemble))
auc_weighted=np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_weighted))
auc_arbitrators=np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_arbitrators))
save_table3(method="Supervised Pretraining - Ensemble",chestray=with_chestray,model=basemodel.name,
            accuracy=accuracy,auc=auc,details=details_,
            other=json.dumps({'weighted':[accuracy_weighted,auc_weighted],
                              'acc_cons_infil':accuracy_consolidation_infiltrates,
                              'arbitrators':[accuracy_arbitrators,auc_arbitrators]}))
save_table_reviewers(method="Supervised Pretraining - Ensemble",chestray=with_chestray,model=basemodel.name,accuracy=rev_acc,auc=rev_auc,details=details_)

if display: perch_confusion_matrix(perch.test['labels'].values,pred_ensemble_weighted.argmax(axis=1))



predictions=pd.DataFrame({'pred':pred_ensemble_weighted.argmax(axis=1),
                          'true':perch.test['labels'],
                          'site':perch.test['SITE'],
                          'age':perch.test['_AGEM'],
                          'reader1':perch.test['REV1'],
                          'reader2':perch.test['REV2'],})
predictions['correct']=predictions['true']==predictions['pred']
predictions['agreement']=predictions['reader1']==predictions['reader2']
predictions['agecat']=predictions['age'].map(lambda x: "(0-12)" if x<12 else "[12-60)")
predictions.groupby('site').agg({'correct':np.mean})
predictions.groupby('agecat').agg({'correct':np.mean})
from statsmodels.nonparametric.smoothers_lowess import lowess
y=lowess(predictions['correct']*1.0,predictions['age'],)
y2=lowess(predictions['agreement']*1.0,predictions['age'],)


plt.plot(y[:,0],y[:,1])
plt.xlabel("Age in months")
plt.ylabel("Classification accuracy")
plt.show()

plt.plot(y2[:,0],y2[:,1])
plt.xlabel("Age in months")
plt.ylabel("Reader agreement (%)")
plt.show()

for s in predictions['site'].unique():
    subset = predictions[predictions['site'] == s]

    # Draw the density plot
    sns.distplot(subset['age'], hist=False, kde=True,
                 kde_kws={'linewidth': 3},
                 label=s)

# Plot formatting
plt.legend(prop={'size': 16}, title='Site')
plt.xlabel('Age (months)')
plt.ylabel('Density')
plt.show()