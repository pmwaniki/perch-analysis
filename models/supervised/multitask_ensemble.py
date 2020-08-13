import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
from models.supervised.applications import MultitaskEnsemble
from models.supervised.applications import (resnet50v2,resnet50,resnet101,densnet121,inceptionv3,
                                            vgg19)
from settings import perch_config,chestray_config,prediction_dir,result_dir
from utils import save_table3
import os,json
from metrics import multilabel_auc,multiclass_acc,consolidation_infiltrates_acc
import matplotlib.pyplot as plt
import pickle
from data.datasets import perch,chestray
from data.generators import multitask_ensemble_generator
from sklearn.metrics import confusion_matrix
from data.preprocess_image import (load_and_preprocess_image,
                                   flip,rotate,
                                   color,
                                   load_and_preprocess_multitask_ensemble,
                                   crop_and_pad,
                                    augment_ensemble,
                                   augument_multitask_ensemble)
import numpy as np
import pandas as pd
from functools import partial
from callbacks import lr_schedule
from models.supervised.multitask_ensemble_utils import train_one_batch
AUTOTUNE=tf.data.experimental.AUTOTUNE
display=os.environ.get("DISPLAY",None)



basemodel=densnet121
weights_='imagenet'
warm_start=False
pooling_='avg'
norm = 0.05
l2_=0.05
l2_embed=0.0001
# dropout_=0.05
crop_prop=0.9
initial_lr=0.1
M=3
epochs_M=50

if weights_ not in [None, 'imagenet'] : raise Exception("Unrecognised weight initialization")

details_="pooling: {} | l2: {:.5f} | l2_emb: {:.6f} | crop: {:.2f} | weights: {}_lr {:.2f}".format(pooling_,l2_,
                                                                                            l2_embed,
                                                                                            crop_prop,
                                                                                            weights_,initial_lr)

weight_path="clr_multitask_ensemble_{}_l2 {:.5f}_crop {:.2f} _ weights {}_lr {:.2f}.pkl".format(basemodel.name,
                                                                                             l2_,crop_prop,weights_,
                                                                                               initial_lr)
input_shape=basemodel.input_shape

model=MultitaskEnsemble(base_model=basemodel,weights=weights_,pooling=pooling_,l2=l2_,
                        l2_embeddings=l2_embed,
                        # drop_out=dropout_
                        )
# model.build([(None,)+input_shape,(None,)+input_shape])
if warm_start: model.load_weights()

perch_train=perch.train_long
perch_test=perch.test_long

path_train_perch=perch_train['path']
labs_train_perch=pd.get_dummies(perch_train['rev_label']).values
rev_train=perch_train['reviewer']
correct_train=(perch_train['labels']==perch_train['rev_label']).values*1.0


path_test_perch=perch_test['path']
labs_test_perch=pd.get_dummies(perch_test['rev_label']).values
rev_test=perch_test['reviewer']
correct_test=(perch_test['labels']==perch_test['rev_label']).values*1.0



# path_train_perch,labs_train_perch,path_test_perch,labs_test_perch=perch.load_data()
path_train_chestray,labs_train_chestray,path_test_chestray,labs_test_chestray=chestray.load_data()



BATCH_SIZE=24

train_dataset=tf.data.Dataset.from_generator(multitask_ensemble_generator,
                                             (tf.string,tf.float32,tf.int32,tf.string,tf.float32),
                                             args=(path_train_perch,
                                                    labs_train_perch,
                                                   rev_train,
                                                    path_train_chestray,
                                                    labs_train_chestray))






load_and_preprocess_multitask_ensemble_=partial(load_and_preprocess_multitask_ensemble,shape=input_shape)
ds_train=train_dataset.map(load_and_preprocess_multitask_ensemble_,num_parallel_calls=AUTOTUNE)


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

augument_multitask_ensemble_=partial(augument_multitask_ensemble,aug_list=augumentations)

ds_train=ds_train.map(augument_multitask_ensemble_,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# ss=[(a,b,c,d) for a,b,c,d in ds_train.take(1)]


image_train_perch=tf.data.Dataset.from_tensor_slices(path_train_perch)
image_test_perch=tf.data.Dataset.from_tensor_slices(path_test_perch)
image_train_chestray=tf.data.Dataset.from_tensor_slices(path_train_chestray)
image_test_chestray=tf.data.Dataset.from_tensor_slices(path_test_chestray)

rev_train_perch=tf.data.Dataset.from_tensor_slices(rev_train)
rev_test_perch=tf.data.Dataset.from_tensor_slices(rev_test)

label_train_perch=tf.data.Dataset.from_tensor_slices(tf.cast(labs_train_perch,tf.float32))
label_test_perch=tf.data.Dataset.from_tensor_slices(tf.cast(labs_test_perch,tf.float32))
label_train_chestray=tf.data.Dataset.from_tensor_slices(tf.cast(labs_train_chestray,tf.float32))
label_test_chestray=tf.data.Dataset.from_tensor_slices(tf.cast(labs_test_chestray,tf.float32))


load_and_preprocess_image_=partial(load_and_preprocess_image,shape=input_shape)
image_train_ds_perch=image_train_perch.map(load_and_preprocess_image_,
                               num_parallel_calls=AUTOTUNE).\
    map(partial(tf.subtract,y=0.5),num_parallel_calls=AUTOTUNE).\
    map(partial(tf.multiply,y=2.0),num_parallel_calls=AUTOTUNE)
image_test_ds_perch=image_test_perch.map(load_and_preprocess_image_,
                             num_parallel_calls=AUTOTUNE).\
    map(partial(tf.subtract,y=0.5),num_parallel_calls=AUTOTUNE).\
    map(partial(tf.multiply,y=2.0),num_parallel_calls=AUTOTUNE)

image_train_ds_chestray=image_train_chestray.map(load_and_preprocess_image_,
                              num_parallel_calls=AUTOTUNE).\
    map(partial(tf.subtract,y=0.5),num_parallel_calls=AUTOTUNE).\
    map(partial(tf.multiply,y=2.0),num_parallel_calls=AUTOTUNE)
image_test_ds_chestray=image_test_chestray.map(load_and_preprocess_image_,
                             num_parallel_calls=AUTOTUNE).\
    map(partial(tf.subtract,y=0.5),num_parallel_calls=AUTOTUNE).\
    map(partial(tf.multiply,y=2.0),num_parallel_calls=AUTOTUNE)

# augumentations=[
#     flip,
#     rotate,
#     color,
#     #crop_and_pad,
# ]

# for aug in augumentations:
#     image_train_ds_perch=image_train_ds_perch.map(aug)
#     image_train_ds_chestray=image_train_ds_chestray.map(aug)



ds_train_perch = tf.data.Dataset.zip((tf.data.Dataset.zip((image_train_ds_perch,rev_train_perch)),label_train_perch))
ds_test_perch= tf.data.Dataset.zip((tf.data.Dataset.zip((image_test_ds_perch,rev_test_perch)),label_test_perch )).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

ds_train_chestray = tf.data.Dataset.zip((image_train_ds_chestray,label_train_chestray))
ds_test_chestray= tf.data.Dataset.zip((image_test_ds_chestray,label_test_chestray )).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)




ds_train_perch=ds_train_perch.batch(BATCH_SIZE)
ds_train_perch=ds_train_perch.prefetch(buffer_size=AUTOTUNE)


ds_train_chestray=ds_train_chestray.batch(BATCH_SIZE)
ds_train_chestray=ds_train_chestray.prefetch(buffer_size=AUTOTUNE)





def eval_model(model_,data,metrics={'auc':multilabel_auc,'acc':multiclass_acc}):
    predictions=[(model_(images),labs) for images,labs in data]
    preds=[p for p,l in predictions]
    labs=[l for p,l in predictions]
    preds=np.concatenate(preds,axis=0)
    labs=np.concatenate(labs,axis=0)
    metrics2={m:fun(labs,preds) for m,fun in metrics.items()}
    return metrics2

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







optimizer_perch = tf.keras.optimizers.SGD(lr=initial_lr)
optimizer_chestray = tf.keras.optimizers.SGD(lr=initial_lr)



lr_list,lr_index=lr_schedule(initial_lr,M=M,size=epochs_M)
print(f"Learning rate minimum: {np.min(lr_list)}")

epochs = M*epochs_M

loss_perch = []
loss_chestray = []
auc_perch = []
auc_chestray = []
acc_perch = []
acc_chestray = []
weights_perch=[]
weights_chestray=[]
# model.freeze_conv_layers()
if warm_start:
    with open(os.path.join(result_dir,"finetune",weight_path),"rb") as f:
        saved_weights=pickle.load(f)
    weights_perch,weights_chestray=saved_weights
    model.model_perch.set_weights(weights_perch[-1])
    model.model_chestray.set_weights(weights_chestray[-1])

train_fun=tf.function(train_one_batch)
ep_start=0
for epoch in range(ep_start,epochs):
    print(f'Epoch {epoch}: Setting learning rate to {lr_list[epoch]}')
    tf.keras.backend.set_value(optimizer_perch.lr, lr_list[epoch] * 0.1)
    tf.keras.backend.set_value(optimizer_chestray.lr, lr_list[epoch])

    for perch_x, perch_y,rev_x, chestray_x, chestray_y in ds_train:
        gradients_perch, gradients_chestray, losses_perch, losses_chestray=train_fun(model, perch_x, rev_x, perch_y,
                                                                                     chestray_x, chestray_y, norm,
                                                                                     optimizer_perch,
                                                                                     optimizer_chestray)
        if np.isnan(losses_chestray.numpy()) | np.isnan(losses_perch.numpy()):
            raise Exception("NAN loss!!!")

    m_perch_train = eval_model(model.model_perch, ds_train_perch, metrics={'auc': multilabel_auc,
                                                                           'acc': categorical_acc,
                                                                           'loss': categorical_loss})
    m_chestray_train = eval_model(model.model_chestray, ds_train_chestray.take(100), metrics={'auc': multilabel_auc,
                                                                                              'acc': binary_acc,
                                                                                              'loss': binary_loss})

    m_perch_test = eval_model(model.model_perch, ds_test_perch, metrics={'auc': multilabel_auc,
                                                                         'acc': categorical_acc,
                                                                         'loss': categorical_loss})
    m_chestray_test = eval_model(model.model_chestray, ds_test_chestray.take(100), metrics={'auc': multilabel_auc,
                                                                                            'acc': binary_acc,
                                                                                            'loss': binary_loss})
    print('Perch:loss: %.4f| val_loss : %.4f| acc: %.2f| val_acc: %.2f| auc: %.2f| val_auc: %.2f' % (
        m_perch_train['loss'],
        m_perch_test['loss'],
        m_perch_train['acc'],
        m_perch_test['acc'],
        np.mean(m_perch_train['auc']),
        np.mean(m_perch_test['auc'])))
    print('Chestray:loss: %.4f| val_loss : %.4f| acc: %.2f| val_acc: %.2f| auc: %.2f| val_auc: %.2f' % (
        m_chestray_train['loss'],
        m_chestray_test['loss'],
        m_chestray_train['acc'],
        m_chestray_test['acc'],
        np.mean(m_chestray_train['auc']),
        np.mean(m_chestray_test['auc'])))

    loss_perch.append((m_perch_train['loss'], m_perch_test['loss']))
    acc_perch.append((m_perch_train['acc'], m_perch_test['acc']))
    auc_perch.append((np.mean(m_perch_train['auc']), np.mean(m_perch_test['auc'])))

    loss_chestray.append((m_chestray_train['loss'], m_chestray_test['loss']))
    acc_chestray.append((m_chestray_train['acc'], m_chestray_test['acc']))
    auc_chestray.append((np.mean(m_chestray_train['auc']), np.mean(m_chestray_test['auc'])))
    if epoch in lr_index:
        weights_perch.append([w.numpy() for w in model.model_perch.weights])
        weights_chestray.append([w.numpy() for w in model.model_chestray.weights])
    # improving = checkpoint.compare(m_perch_test['loss'])
    # if not improving: break

with open(os.path.join(result_dir,"finetune",weight_path),"wb") as f:
    pickle.dump([weights_perch,weights_chestray],f)

fig, axs = plt.subplots(3, 2, figsize=(8, 6))
for row, met in enumerate(["loss", "auc", "acc"]):
    for col, d in enumerate(['perch', 'chestray']):
        dd = eval("_".join([met, d]))
        axs[row][col].plot([tr for tr, ts in dd], label='train')
        axs[row][col].plot([ts for tr, ts in dd], label='test', linestyle='dashed')
        # axs[row][col].legend()
        axs[row][col].set_title(d)
        axs[row][col].set_ylabel(met)
if display:
    plt.show()
else:
    plt.savefig("/home/pmwaniki/Dropbox/tmp/multitask_ensemble_%s_%s.png" % (basemodel.name,details_))

# model.save_weights()

embeddings=[i for i in model.weights if 'embedding' in i.name][0].numpy()
plt.matshow(embeddings)
plt.colorbar()
if display: plt.show()



#Predict ensemble
pred_all=[]
final_test_im=tf.data.Dataset.from_tensor_slices(perch.test['path']).map(load_and_preprocess_image_).\
    map(partial(tf.subtract,y=0.5),num_parallel_calls=AUTOTUNE).\
    map(partial(tf.multiply,y=2.0),num_parallel_calls=AUTOTUNE)
for i in range(18):
    print("Making prediction for reviewer %d ..." % (i+1))
    rev_ = tf.data.Dataset.from_tensor_slices([i] * perch.test.shape[0])
    ds_test_ = tf.data.Dataset.zip((final_test_im, rev_)).batch(50).prefetch(AUTOTUNE)
    pred_ens = []
    for w in weights_perch:
        model.model_perch.set_weights(w)
        p=[model.model_perch.predict(k) for k in ds_test_]
        pred_ens.append(np.concatenate(p,axis=0))
    pred_ens2=np.stack(pred_ens)
    pred_all.append(pred_ens2.mean(axis=0))




[multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_all[i]) for i in range(18)]
[np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_all[i])) for i in range(18)]

rev_acc=perch_train.groupby('reviewer').apply(lambda df: np.mean(df['labels']==df['rev_label'])).to_dict()
rev_weights={k:v/np.sum(list(rev_acc.values())) for k,v in rev_acc.items() }
rev_weights=[rev_weights[i] for i in range(18)]


# aggregate prediction
pred_stacked=np.stack(pred_all)
pred_ensemble=pred_stacked.mean(axis=0)
pred_ensemble_radiologists=pred_stacked[14:,:,:].mean(axis=0)
pred_ensemble_weighted=np.average(pred_stacked,axis=0,weights=np.array(rev_weights))
# pred_ensemble_weighted2=np.average(pred_stacked,axis=0,weights=rev_weightsb2)
accuracy=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble)
accuracy_consolidation_infiltrates=consolidation_infiltrates_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble)
accuracy_radiologists=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_radiologists)
accuracy_weighted=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_weighted)
# accuracy_weighted2=multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_weighted2)

auc=np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_ensemble))
auc_weighted=np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_weighted))
# auc_weighted2=np.mean(multilabel_auc(pd.get_dummies(perch.test['labels']).values,pred_ensemble_weighted2))
#majority vote
pred_ensemble_cat=pred_stacked.argmax(axis=2)
pred_cat_prob=np.zeros((len(perch.test['labels']),5))
for p in range(pred_cat_prob.shape[0]):
    for c in range(5):
        pred_cat_prob[p,c]=np.mean(pred_ensemble_cat[:,p]==c)

multiclass_acc(pd.get_dummies(perch.test['labels']).values,pred_cat_prob)




save_table3(method="Multi-task learning:Ensemble",
            chestray=True,
            model=basemodel.name,
            accuracy=accuracy,
            auc=auc,
            details=details_,
            other=json.dumps({'weighted':[accuracy_weighted,auc_weighted],
                              # 'weighted_model':[accuracy_weighted2,auc_weighted2],
                              'acc_cons_infil':accuracy_consolidation_infiltrates}))

