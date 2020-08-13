import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
from models.supervised.applications import ChestnetModel
from models.supervised.applications import vgg16,resnet50,densnet121,inceptionv3,vgg19,resnet50v2
import matplotlib.pyplot as plt
import pickle
from settings import result_dir
from data.datasets import chestray
from data.preprocess_image import load_and_preprocess_image,flip,rotate, color,crop_and_pad,plot_images
from sklearn.utils import class_weight
import numpy as np
import os
from functools import partial
from metrics import multilabel_auc
from callbacks import CyclicLR
from metrics import multilabel_auc,multilabel_acc
AUTOTUNE=tf.data.experimental.AUTOTUNE
display=os.environ.get("DISPLAY",None)


basemodel=densnet121
weights_= 'imagenet'
pooling_='avg'
warm_start=False
input_shape=basemodel.input_shape
l2_=0.0005
# dropout_=0.5
crop_prop=0.5
initial_lr=0.1
M=6;size=20
weight_path="clr_chestnet_{}_l2 {:.5f}_crop {:.2f} _ weights {}_lr {:.2f}.pkl".format(basemodel.name,
                                                                                             l2_,crop_prop,weights_,
                                                                                               initial_lr)

model=ChestnetModel(base_model=basemodel,n_out=15,pooling=pooling_,weights=weights_,
                    # drop_out=dropout_,
                    l2=l2_)
model.build((None,)+input_shape)

path_train,labs_train,path_test,labs_test=chestray.load_data()




BATCH_SIZE=24

image_train=tf.data.Dataset.from_tensor_slices(path_train)
image_test=tf.data.Dataset.from_tensor_slices(path_test)

label_train=tf.data.Dataset.from_tensor_slices(tf.cast(labs_train,tf.float32))
label_test=tf.data.Dataset.from_tensor_slices(tf.cast(labs_test,tf.float32))


image_train_ds=image_train.map(lambda x:load_and_preprocess_image(x,shape=input_shape),
                               num_parallel_calls=AUTOTUNE)
image_test_ds=image_test.map(lambda x:load_and_preprocess_image(x,shape=input_shape),
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





ds_train=ds_train.shuffle(buffer_size=5000)
ds_train=ds_train.repeat()
ds_train=ds_train.batch(BATCH_SIZE)
ds_train=ds_train.prefetch(buffer_size=AUTOTUNE)

ds_test=ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)




optimizer=tf.keras.optimizers.SGD(lr=initial_lr,clipnorm=0.1,momentum=0.9)


swa_callback=CyclicLR(a0=initial_lr,M=M,size=size)
print("Minimum learning rate: %7.6f" % np.min(swa_callback.lr_list))

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.binary_crossentropy,
              metrics=["accuracy"])
if warm_start:
    with open(os.path.join(result_dir, "finetune", weight_path), "rb") as f:
        saved_weights=pickle.load( f)
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

plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
if display:
    plt.show()
else:
    plt.savefig("/home/pmwaniki/Dropbox/tmp/chestnet_%s_crop_%.2f_l2_%.4f.png" % (basemodel.name,crop_prop,l2_))

pred_ens=[]
for w in swa_callback.weight_list:
    model.set_weights(w)
    pred_ens.append(model.predict(ds_test))
pred_ens=np.stack(pred_ens)




weights_swa=[]
for w2 in range(len(model.weights)):
    w2_l=[w[w2] for w in swa_callback.weight_list[3:]]
    w2_stacked=np.stack(w2_l)
    w2_agg=w2_stacked.mean(axis=0)
    weights_swa.append(w2_agg)
model.set_weights(weights_swa)
pred_ens_swa=model.predict(ds_test)

pred_ens_stacked=pred_ens.mean(axis=0)
sep_ens_acc=[multilabel_acc(labs_test,pred_ens[i,:,:]) for i in range(pred_ens.shape[0])]
sep_ens_auc=[np.mean(multilabel_auc(labs_test,pred_ens[i,:,:])) for i in range(pred_ens.shape[0])]

best_model=np.where(sep_ens_auc==np.max(sep_ens_auc))[0][0]
model.set_weights(swa_callback.weight_list[best_model])

auc=np.mean(multilabel_auc(labs_test,pred_ens_stacked))
auc_swa=np.mean(multilabel_auc(labs_test,pred_ens_swa))
acc=multilabel_acc(labs_test,pred_ens_stacked)

multilabel_auc(labs_test,pred_ens_swa)
multilabel_acc(labs_test,pred_ens_swa)






model.save_weights()

