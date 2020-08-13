import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

from models.unsupervised.vae import (CVAE,ClassifierEnsemble,generate_and_save_images,plot_reconstruction)
from utils import save_table3
from metrics import multilabel_auc,multiclass_acc,consolidation_infiltrates_acc
from scipy.cluster.hierarchy import linkage,dendrogram
from callbacks import CyclicLR
from data.preprocess_image import (
    flip,
    rotate,
    color,
    crop_and_pad,
)

from data.datasets import perch
from settings import result_dir,dropbox_dir
import os
import json,pickle
import pandas as pd
from functools import partial
from data.preprocess_image import load_and_preprocess_image_ensemble,augment_ensemble
AUTOTUNE=tf.data.experimental.AUTOTUNE
display=os.environ.get("DISPLAY",None)


input_shape=(224,224,3)
l2_=0.0005
crop_prop=0.9
sigmoid_loss=False
warm_start = False
init_weights="imagenet"
data_name="chestray"
loss_type="elbo"
latent_dim=1024 # 128, 1024
beta=1 #0,1, 5,
type_="VAE"
if loss_type=="mmd": type_="Info VAE"
if (beta > 1) & (loss_type=="elbo"): type_="Beta VAE"
if beta==0 : type_="Auto-encoder"
initial_lr=0.1
M=6;size=50
weight_path=os.path.join(result_dir,"embeddings/VAE_{}_{}_z{}_b{}_w_{}_weights.ckpt".\
                         format(data_name,loss_type,latent_dim,beta,init_weights))
save_path=os.path.join(result_dir,
                    "embeddings/finetune_ensemble_{}_{}_z{}_b{}.ckpt".format(data_name,loss_type,latent_dim,beta))


details_="{}_z{}_b{}_l2{:.6f}_crop{:.1f}".format(loss_type,latent_dim,beta,l2_,crop_prop)
ens_weight_path=os.path.join(result_dir,
                             "embeddings/vae_finetune_ensemble_{}_{}_z{}_b{}_l2{:.6f}_crop{:.2f}.pkl".format(data_name,loss_type,latent_dim,beta,l2_,crop_prop))

optimizer=tf.keras.optimizers.Adam(1e-4,clipnorm=1.0)

batch_size=24


model=CVAE(latent_dim,input_shape=input_shape,loss=loss_type,layers_decoder=5,l2=l2_)
model.load_weights(weight_path)

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




image_train=tf.data.Dataset.from_tensor_slices((path_train,rev_train))
image_test=tf.data.Dataset.from_tensor_slices((path_test,rev_test))

label_train=tf.data.Dataset.from_tensor_slices(tf.cast(labs_train,tf.float32))
label_test=tf.data.Dataset.from_tensor_slices(tf.cast(labs_test,tf.float32))



load_and_preprocess_image_ensemble_=partial(load_and_preprocess_image_ensemble,shape=input_shape)

image_train_ds=image_train.map(load_and_preprocess_image_ensemble_,
                               num_parallel_calls=AUTOTUNE)
image_test_ds=image_test.map(load_and_preprocess_image_ensemble_,
                             num_parallel_calls=AUTOTUNE)

augmentations=[
    partial(color,cont_lower=0.3,cont_upper=0.99,bright_delta=0.01),
    partial(crop_and_pad,proportion=crop_prop,width=.5,height=.5),
    flip,
    rotate,
    partial(tf.subtract,y=0.5),
    partial(tf.multiply,y=2.0),


]

augmentations_test=[
    partial(tf.subtract,y=0.5),
    partial(tf.multiply,y=2.0),
]

def undo_center_scale(x):
    x=x/2.0
    return x+0.5


augment_ensemble_=partial(augment_ensemble,aug_funs=augmentations)
augment_test_ensemble_=partial(augment_ensemble,aug_funs=augmentations_test)

image_train_ds=image_train_ds.map(augment_ensemble_)
image_test_ds=image_test_ds.map(augment_test_ensemble_)

ss=[im for im,rev in image_train_ds.batch(25).take(1)][0]
ss2=np.zeros((input_shape[0]*5,input_shape[1]*5,3))
k=0
for i in range(5):
    for j in range(5):
        ss2[i*input_shape[0]:(i+1)*input_shape[0],j*input_shape[1]:(j+1)*input_shape[1]]=ss[k,:,:]
        k=k+1
plt.imshow(undo_center_scale(ss2))
if display: plt.show()


ds_train = tf.data.Dataset.zip((image_train_ds,label_train))
ds_test= tf.data.Dataset.zip((image_test_ds,label_test ))





ds_train=ds_train.shuffle(buffer_size=4500)
ds_train=ds_train.repeat()
ds_train=ds_train.batch(batch_size)
ds_train=ds_train.prefetch(buffer_size=AUTOTUNE)

ds_test=ds_test.batch(batch_size).prefetch(AUTOTUNE)




model2=ClassifierEnsemble(model,l2=l2_,l2_embeddings=0.0001)
if warm_start:model2.load_weights(save_path)
model2.build([(batch_size,)+input_shape,(batch_size,1)])




model2.compile(optimizer=tf.keras.optimizers.SGD(lr=initial_lr),
               loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
               metrics=['accuracy'])
swa_callback=CyclicLR(a0=initial_lr,M=M,size=size)
if warm_start:
    with open(ens_weight_path,"rb") as f:
        saved_weights=pickle.load(f)
    swa_callback.weight_list=saved_weights
    model2.set_weights(saved_weights[-1])
model2.fit(ds_train,
          epochs=M*size,
          steps_per_epoch=int(len(path_train)/batch_size),
          validation_data=ds_test,validation_steps=int(len(path_test)/batch_size),
          callbacks=[swa_callback],
          verbose=1)

with open(ens_weight_path,"wb") as f:
    pickle.dump(swa_callback.weight_list,f)

fig,axs=plt.subplots(nrows=1,ncols=2)
axs[0].plot(model2.history.history['loss'])
axs[0].plot(model2.history.history['val_loss'])
axs[0].set_title("loss")
axs[0].set_ylim(top=3.,bottom=0)
axs[1].plot(model2.history.history['accuracy'])
axs[1].plot(model2.history.history['val_accuracy'])
axs[1].set_title("accuracy")
if display:
    plt.show()
else:
    plt.savefig("/home/pmwaniki/Dropbox/tmp/finetune_ensemble_%s_%s.png" % (data_name,details_))



model2.save_weights(save_path)



## embeddings
embeddings=[i for i in model2.weights if 'embedding' in i.name][0].numpy()
plt.matshow(embeddings)
plt.colorbar()
if display: plt.show()

#dendogram
linked = linkage(embeddings, 'complete',metric='correlation')

labelList = range(1, 19)

plt.figure(figsize=(10, 7))
dendrogram(linked,
            orientation='top',
            labels=labelList,
            distance_sort='descending',
            show_leaf_counts=True)
if display: plt.show()

#reviewer accuracy
rev_acc=perch_train.groupby('reviewer').apply(lambda df: np.mean(df['labels']==df['rev_label'])).to_dict()
rev_weights={k:v/np.sum(list(rev_acc.values())) for k,v in rev_acc.items() }
rev_weights=[rev_weights[i] for i in range(18)]



pred_all=[]
for i in range(18):
    print("Making prediction for reviewer %d ..." % (i+1))
    ds_test_=tf.data.Dataset.from_tensor_slices((perch.test['path'],[i]*perch.test.shape[0])).\
        map(load_and_preprocess_image_ensemble_).map(augment_test_ensemble_).batch(20).prefetch(AUTOTUNE)
    pred_ens = []
    for w in swa_callback.weight_list:
        model2.set_weights(w)
        p=[model2.predict(k) for k in ds_test_]
        pred_ens.append(np.concatenate(p,axis=0))
    pred_ens2=np.stack(pred_ens)
    pred_all.append(pred_ens2.mean(axis=0))
#accuracy for each rater
rev_acc=[multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_all[i]) for i in range(18)]
rev_auc=[np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_all[i])) for i in range(18)]
# aggregate prediction
pred_stacked=np.stack(pred_all)




pred_ensemble=pred_stacked.mean(axis=0)
pred_ensemble_weighted=np.average(pred_stacked,axis=0,weights=np.array(rev_weights))
pred_ensemble_arbitrators=pred_stacked[14:,:,:].mean(axis=0)
accuracy=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble)
accuracy_weighted=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_weighted)
accuracy_arbitrators=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_arbitrators)
auc=np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_ensemble))
auc_weighted=np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_weighted))
auc_arbitrators=np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_arbitrators))
accuracy_consolidation_infiltrates=consolidation_infiltrates_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble)

save_table3(method="Unsupervised Pretraining:Ensemble",chestray=False if data_name == "perch" else True,
            model=type_,accuracy=accuracy,auc=auc,details=details_,
            other=json.dumps({'weighted':[accuracy_weighted,auc_weighted],
                              # 'weighted_model':[accuracy_weighted2,auc_weighted2],
                              'acc_cons_infil':accuracy_consolidation_infiltrates,
                              'arbitrators':[accuracy_arbitrators,auc_arbitrators]}))
