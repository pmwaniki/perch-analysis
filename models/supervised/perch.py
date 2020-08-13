import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
from models.supervised.applications import PerchModel
from models.supervised.applications import vgg16,resnet50,densnet121,inceptionv3,vgg19,resnet50v2
from settings import perch_config,result_dir
from utils import save_table3
import os
from metrics import multilabel_auc,multiclass_acc,consolidation_infiltrates_acc,perch_confusion_matrix
import matplotlib.pyplot as plt
import pickle,json
from data.datasets import perch

display=os.environ.get("DISPLAY",None)

from data.preprocess_image import (load_and_preprocess_image,
                                   flip,rotate,color,crop_and_pad,
                                   crop_and_resize,plot_images,divide,
                                   subtract)

import numpy as np
from callbacks import CyclicLR
from functools import partial

AUTOTUNE=tf.data.experimental.AUTOTUNE



basemodel=inceptionv3
weights_= 'imagenet' #chestnet, imagenet None
init_weights='imagenet'
warm_start=False
label_smoothening=True
pooling_='avg'
l2_=0.0005 #0.0005, 0.0005
crop_prop=0.5# 0.2,0.5, 0.9
initial_lr=0.1
M=6;size=50
if weights_ not in [None, 'imagenet', 'chestnet'] : raise Exception("Unrecognised weight initialization")
with_chestray=True if weights_=='chestnet' else False

details_="pooling: {} | l2: {:.5f} | crop: {:.2f} | weights: {} | lr: {:.2f}".format(pooling_,l2_,crop_prop,weights_,initial_lr)
weight_path="clr_{}_l2 {:.5f}_crop {:.2f} _ weights {}_lr {:.2f}.pkl".format(basemodel.name,
                                                                                             l2_,crop_prop,weights_,
                                                                                               initial_lr)
if weights_ == "chestnet":
    weight_path = weight_path.replace(".pkl", "_initw_{}.pkl".format(init_weights))
    details_ = details_ + "_initw{}".format(init_weights)

if label_smoothening:
    weight_path=weight_path.replace(".pkl","_smooth.pkl")
    details_=details_ + "_smooth"

input_shape=basemodel.input_shape

model=PerchModel(base_model=basemodel,n_out=5,weights=weights_,
                 pooling=pooling_,l2=l2_,
                 init_weights=init_weights)
model.build((None,)+input_shape)

path_train,labs_train,path_test,labs_test=perch.load_data()



BATCH_SIZE=32

image_train=tf.data.Dataset.from_tensor_slices(path_train)
image_test=tf.data.Dataset.from_tensor_slices(path_test)

label_train=tf.data.Dataset.from_tensor_slices(tf.cast(labs_train,tf.float32))
label_test=tf.data.Dataset.from_tensor_slices(tf.cast(labs_test,tf.float32))


load_and_preprocess_image_=partial(load_and_preprocess_image,shape=input_shape)

image_train_ds=image_train.map(load_and_preprocess_image_,
                               num_parallel_calls=AUTOTUNE)
image_test_ds=image_test.map(load_and_preprocess_image_,
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

for aug in augumentations:
    image_train_ds=image_train_ds.map(aug)
for aug_test in augumentations_test:
    image_test_ds=image_test_ds.map(aug_test)

if display: plot_images(dataset=image_train_ds.take(5),n_images=5,samples_per_image=5,shape=input_shape)

ds_train = tf.data.Dataset.zip((image_train_ds,label_train))
ds_test= tf.data.Dataset.zip((image_test_ds,label_test ))




ds_train=ds_train.shuffle(buffer_size=4500)
ds_train=ds_train.repeat()
ds_train=ds_train.batch(BATCH_SIZE)
ds_train=ds_train.prefetch(buffer_size=AUTOTUNE)

ds_test=ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)

swa_callback=CyclicLR(a0=initial_lr,M=M,size=size)
print("Minimum learning rate: %7.6f" % np.min(swa_callback.lr_list))

if warm_start:
    with open(os.path.join(result_dir, "finetune", weight_path), "rb") as f:
        saved_weights=pickle.load( f)
    model.set_weights(saved_weights[-1])
    swa_callback.weight_list=saved_weights



optimizer=tf.keras.optimizers.SGD(lr=initial_lr)

if label_smoothening:
    model.compile(optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  metrics=["accuracy"])
else:
    model.compile(optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=["accuracy"])



model.fit(ds_train,
          epochs=M*size,
          steps_per_epoch=int(len(path_train)/BATCH_SIZE),
          validation_data=ds_test,validation_steps=int(len(path_test)/BATCH_SIZE),
          callbacks=[swa_callback],
          verbose=1
          )

fig,axs=plt.subplots(nrows=1,ncols=2)
axs[0].plot(model.history.history['loss'])
axs[0].plot(model.history.history['val_loss'])
axs[0].set_title("loss")
axs[0].set_ylim(top=3.,bottom=0)
axs[1].plot(model.history.history['accuracy'])
axs[1].plot(model.history.history['val_accuracy'])
axs[1].set_title("accuracy")
if display:
    plt.show()
else:
    plt.savefig("/tmp/perch_%s_%s.png" % (basemodel.name,details_))



with open(os.path.join(result_dir,"finetune",weight_path),"wb") as f:
    pickle.dump(swa_callback.weight_list,f)





#predict ensemble
pred_ens=[]
for w in swa_callback.weight_list:
    model.set_weights(w)
    pred_ens.append(model.predict(ds_test))
pred_ens=np.stack(pred_ens)




weights_swa=[]
for w2 in range(len(model.weights)):
    w2_l=[w[w2] for w in swa_callback.weight_list]
    w2_stacked=np.stack(w2_l)
    w2_agg=w2_stacked.mean(axis=0)
    weights_swa.append(w2_agg)
model.set_weights(weights_swa)
pred_ens_swa=model.predict(ds_test)

pred_ens_stacked=pred_ens.mean(axis=0)
sep_ens_acc=[multiclass_acc(labs_test,pred_ens[i,:,:]) for i in range(pred_ens.shape[0])]
sep_ens_auc=[np.mean(multilabel_auc(labs_test,pred_ens[i,:,:])) for i in range(pred_ens.shape[0])]

auc=np.mean(multilabel_auc(labs_test,pred_ens_stacked))
acc=multiclass_acc(labs_test,pred_ens_stacked)
cons_infil_scores=consolidation_infiltrates_acc(labs_test,pred_ens_stacked)

multilabel_auc(labs_test,pred_ens_swa)
multiclass_acc(labs_test,pred_ens_swa)


if display:perch_confusion_matrix(labs_test.argmax(axis=1),pred_ens_stacked.argmax(axis=1))


save_table3(method="Supervised Pretraining",chestray=with_chestray,model=basemodel.name,
            accuracy=acc,auc=auc,details=details_,
            other=json.dumps({'acc_cons_infil':cons_infil_scores}))
