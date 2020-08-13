import tensorflow as tf

from models.supervised.applications import Multitask
from models.supervised.applications import (vgg16,resnet50,densnet121,inceptionv3,
                                            resnet50v2)
from settings import perch_config,chestray_config,prediction_dir,result_dir
from utils import save_table3
from callbacks import lr_schedule
import os
from metrics import multilabel_auc,multiclass_acc,consolidation_infiltrates_acc
import matplotlib.pyplot as plt
import pickle,json
from data.datasets import perch,chestray
from data.generators import multitask_generator
from sklearn.metrics import confusion_matrix
from data.preprocess_image import (load_and_preprocess_image,flip,rotate,color,
                                   load_and_preprocess_multitask,crop_and_pad,
                                   crop_and_resize,augument_multitask)
import numpy as np
from functools import partial
from models.supervised.multitask_utils import train_one_step,compute_gradients_chestray,compute_gradients_perch,apply_gradients_perch,apply_gradients_chestray
AUTOTUNE=tf.data.experimental.AUTOTUNE
display=os.environ.get("DISPLAY",None)


basemodel=inceptionv3
weights_='imagenet'
label_smoothening=True
warm_start=False
pooling_='avg'
l2_=0.0005
# dropout_=0.5
crop_prop=0.2 #0.2, 0.5, 0.9 Proportion of images that are cropped and padded
initial_lr=0.1
scale_lr_perch=0.1
M=3
epochs_M=50
norm_=0.05 #0.05, None

if weights_ not in [None, 'imagenet'] : raise Exception("Unrecognised weight initialization")

details_="pooling: {} | l2: {:.5f} | crop: {:.2f} | weights: {} | lr {:.4f}".format(pooling_,l2_,crop_prop,weights_,scale_lr_perch)
weight_path="clr_multitask_{}_l2 {:.5f}_crop {:.2f} _ weights {}_lr {:.4f}.pkl".format(basemodel.name,
                                                                                             l2_,crop_prop,weights_,
                                                                                               scale_lr_perch)
if label_smoothening:
    weight_path=weight_path.replace(".pkl","_smooth.pkl")
    details_=details_ + "_smooth"

input_shape=basemodel.input_shape

model=Multitask(base_model=basemodel,weights=weights_,pooling=pooling_,l2=l2_)
model.build([(None,)+input_shape,(None,)+input_shape])
if warm_start: model.load_weights()


path_train_perch,labs_train_perch,path_test_perch,labs_test_perch=perch.load_data()
path_train_chestray,labs_train_chestray,path_test_chestray,labs_test_chestray=chestray.load_data()



BATCH_SIZE=24

train_dataset=tf.data.Dataset.from_generator(multitask_generator,
                                             (tf.string,tf.float32,tf.string,tf.float32),
                                             args=(path_train_perch,
                                                    labs_train_perch,
                                                    path_train_chestray,
                                                    labs_train_chestray))
preprocess=partial(load_and_preprocess_multitask,shape=input_shape)


augmentations=[
    partial(color,cont_lower=0.3,cont_upper=0.99,bright_delta=0.01),
    partial(crop_and_pad,proportion=crop_prop,width=.5,height=.5),
    # partial(crop_and_resize,width=0.7,height=0.7),
    flip,
    rotate,
    partial(tf.subtract,y=0.5),
    partial(tf.multiply,y=2.0),

]

augmentations_test=[
    partial(tf.subtract,y=0.5),
    partial(tf.multiply,y=2.0),
]


augument_multitask_=partial(augument_multitask,aug_list=augmentations)



ds_train=train_dataset.map(preprocess,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.map(augument_multitask_,num_parallel_calls=AUTOTUNE)
ds_train=ds_train.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)



image_train_perch=tf.data.Dataset.from_tensor_slices(path_train_perch)
image_test_perch=tf.data.Dataset.from_tensor_slices(path_test_perch)
image_train_chestray=tf.data.Dataset.from_tensor_slices(path_train_chestray)
image_test_chestray=tf.data.Dataset.from_tensor_slices(path_test_chestray)

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





ds_train_perch = tf.data.Dataset.zip((image_train_ds_perch,label_train_perch))
ds_test_perch= tf.data.Dataset.zip((image_test_ds_perch,label_test_perch )).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

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

def opt_set_lr(optimizer,lr):
    opt_config=optimizer.get_config()
    opt_config['learning_rate']=lr
    return optimizer.from_config(opt_config)

optimizer_perch = tf.keras.optimizers.SGD(lr=initial_lr)
optimizer_chestray = tf.keras.optimizers.SGD(lr=initial_lr)

lr_list,lr_index=lr_schedule(initial_lr,M=M,size=epochs_M)
print(f"Learning rate minimum: {np.min(lr_list)}")
train_fun=tf.function(train_one_step)

epochs = M*epochs_M

loss_perch = []
loss_chestray = []
auc_perch = []
auc_chestray = []
acc_perch = []
acc_chestray = []
weights_perch=[]
weights_chestray=[]
epoch=0
for epoch in range(epoch,epochs):
    print(f'Epoch {epoch}: Setting learning rate to {lr_list[epoch]}')
    tf.keras.backend.set_value(optimizer_perch.lr,lr_list[epoch]*scale_lr_perch)
    tf.keras.backend.set_value(optimizer_chestray.lr, lr_list[epoch])


    for perch_x, perch_y, chestray_x, chestray_y in ds_train:
        gradients_perch, gradients_chestray, losses_perch, losses_chestray=train_fun(model,perch_x,
                                                                                          perch_y,
                                                                                          chestray_x,
                                                                                          chestray_y,
                                                                                          optimizer_perch,
                                                                                          optimizer_chestray,
                                                                                          norm=norm_,
                                                                                     label_smoothening=label_smoothening)
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
    # improving = checkpoint.compare(m_perch_test['loss'])
    # if not improving: break
    if epoch in lr_index:
        weights_perch.append([w.numpy() for w in model.model_perch.weights])
        weights_chestray.append([w.numpy() for w in model.model_chestray.weights])


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
    plt.savefig("/home/pmwaniki/Dropbox/tmp/multitask_%s_%s.png" % (basemodel.name,details_))

# model.save_weights()


pred_ens=[]
for w in weights_perch:
    model.model_perch.set_weights(w)
    pred_ens.append(model.model_perch.predict(ds_test_perch))
pred_ens=np.stack(pred_ens)

pred_ens_stacked=pred_ens[0:5,:,:].mean(axis=0)
sep_ens_acc=[multiclass_acc(labs_test_perch,pred_ens[i,:,:]) for i in range(pred_ens.shape[0])]
sep_ens_auc=[np.mean(multilabel_auc(labs_test_perch,pred_ens[i,:,:])) for i in range(pred_ens.shape[0])]

auc=np.mean(multilabel_auc(labs_test_perch,pred_ens_stacked))
acc=multiclass_acc(labs_test_perch,pred_ens_stacked)
cons_infil_scores=consolidation_infiltrates_acc(labs_test_perch,pred_ens_stacked)



test_pred_class=pred_ens_stacked.argmax(axis=1)
test_y_class=labs_test_perch.argmax(axis=1)

confusion_matrix(test_y_class,test_pred_class)




save_table3(method="Multi-task learning",
            chestray=True,
            model=basemodel.name,
            accuracy=acc,
            auc=auc,
            details=details_,
            other=json.dumps({'acc_cons_infil':cons_infil_scores}))

