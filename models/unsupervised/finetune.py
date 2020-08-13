import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
import matplotlib.pyplot as plt
import numpy as np


from models.unsupervised.vae import (CVAE,Classifier,generate_and_save_images,plot_reconstruction)
from utils import save_table3
from metrics import multilabel_auc,multiclass_acc

from data.preprocess_image import (
    flip,
    rotate,
    color,
    crop_and_pad,
)

from data.datasets import perch
from settings import result_dir,dropbox_dir
import os
import pickle,json
from functools import partial
from data.preprocess_image import load_and_preprocess_image,plot_images,binarize
AUTOTUNE=tf.data.experimental.AUTOTUNE
display=os.environ.get("DISPLAY",None)
from callbacks import CyclicLR


input_shape=(224,224,3)
l2_=0.05
crop_prop=0.9
label_smoothening=True
sigmoid_loss=False
warm_start = False
data_name="chestray"
init_weights='imagenet'
loss_type="elbo"

latent_dim=1024 #128, 1024
beta=1 #0,1, 5,

type_="VAE"
if loss_type=="mmd": type_="Info VAE"
if (beta > 1) & (loss_type=="elbo"): type_="Beta VAE"
if beta==0 : type_="Auto-encoder"
initial_lr=.1
M=6;size=50
weight_path=os.path.join(result_dir,"embeddings/VAE_{}_{}_z{}_b{}_w_{}_weights.ckpt".\
                         format(data_name,loss_type,latent_dim,beta,init_weights))
save_path=os.path.join(result_dir,
                    "embeddings/finetune_{}_{}_z{}_b{}_w_{}.ckpt".format(data_name,loss_type,latent_dim,beta,init_weights))


details_="{}_z{}_b{}_l2{:.6f}_crop{:.2f}_lr{:.1f}_w_{}".format(loss_type,latent_dim,beta,l2_,crop_prop,initial_lr,init_weights)
ens_weight_path="vae_finetune_{}_{}_z{}_b{}_l2{:.6f}_crop{:.2f}_lr{:.1f}_w_{}.pkl".format(data_name,loss_type,latent_dim,beta,l2_,crop_prop,initial_lr,init_weights)

if label_smoothening:
    ens_weight_path=ens_weight_path.replace(".pkl","_smooth.pkl")
    details_=details_ + "_smooth"


batch_size=32



model=CVAE(latent_dim,input_shape=input_shape,loss=loss_type,layers_decoder=5,l2=l2_)

model.load_weights(weight_path)
path_train,labs_train,path_test,labs_test=perch.load_data()

train_im=tf.data.Dataset.from_tensor_slices(path_train)
test_im=tf.data.Dataset.from_tensor_slices(path_test)
train_labs=tf.data.Dataset.from_tensor_slices(tf.cast(labs_train,tf.float32))
test_labs=tf.data.Dataset.from_tensor_slices(tf.cast(labs_test,tf.float32))

load_and_preprocess_image_=partial(load_and_preprocess_image,shape=input_shape)
train_im_ds=train_im.map(load_and_preprocess_image_,num_parallel_calls=AUTOTUNE)
test_im_ds=test_im.map(load_and_preprocess_image_)

plot_files=[path_test[i] for i in [0,1,5,7,43]]
plot_ds=tf.data.Dataset.from_tensor_slices(plot_files)
plot_ds=plot_ds.map(load_and_preprocess_image_)


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

for aug in augmentations:
    train_im_ds=train_im_ds.map(aug)
for aug in augmentations_test:
    test_im_ds=test_im_ds.map(aug)

train_ds=tf.data.Dataset.zip((train_im_ds,train_labs))
test_ds=tf.data.Dataset.zip((test_im_ds,test_labs))







train_dataset=train_ds.shuffle(len(path_train)).repeat().batch(batch_size).prefetch(buffer_size=AUTOTUNE)
test_dataset=test_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)





model2=Classifier(model,l2=l2_)
model2.build((batch_size,)+input_shape)




if label_smoothening:
    loss_fun=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2)
else:
    loss_fun=tf.keras.losses.categorical_crossentropy

model2.compile(optimizer=tf.keras.optimizers.SGD(lr=initial_lr),
               loss=loss_fun,
               metrics=['accuracy'])
swa_callback=CyclicLR(a0=initial_lr,M=M,size=size)
model2.fit(train_dataset,
          epochs=M*size,
          steps_per_epoch=int(len(path_train)/batch_size),
          validation_data=test_dataset,validation_steps=int(len(path_test)/batch_size),
          callbacks=[swa_callback],
          verbose=1)

with open(os.path.join(result_dir,"embeddings",ens_weight_path),"wb") as f:
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
    plt.savefig("/home/pmwaniki/Dropbox/tmp/finetune_%s_%s.png" % (data_name,details_))


pred_ens=[]
for w in swa_callback.weight_list:
    model2.set_weights(w)
    pred_ens.append(model2.predict(test_dataset))
pred_ens=np.stack(pred_ens)

pred_ens_stacked=pred_ens.mean(axis=0)
sep_ens_acc=[multiclass_acc(labs_test,pred_ens[i,:,:]) for i in range(pred_ens.shape[0])]
sep_ens_auc=[np.mean(multilabel_auc(labs_test,pred_ens[i,:,:])) for i in range(pred_ens.shape[0])]

auc=np.mean(multilabel_auc(labs_test,pred_ens_stacked))
acc=multiclass_acc(labs_test,pred_ens_stacked)





save_table3(method="Unsupervised Pretraining",chestray=False if data_name == "perch" else True,
            model=type_,accuracy=acc,auc=auc,details=details_)