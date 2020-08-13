import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)

from models.unsupervised.vae import (CVAE,
                                     compute_gradients,
                                     apply_gradients,
                                     generate_and_save_images,
                                     compute_loss,train_one_step,
                                     plot_reconstruction)

from data.preprocess_image import (
    flip,
    rotate,

)
from functools import partial
from data.datasets import perch,chestray
from settings import result_dir,dropbox_dir
import os
import pickle
from data.preprocess_image import load_and_preprocess_image,plot_images,binarize
from callbacks import lr_schedule
AUTOTUNE=tf.data.experimental.AUTOTUNE
display=os.environ.get("DISPLAY",None)



input_shape=(224,224,3)
l2_=0.0005
sigmoid_loss=False
warm_start = False
data_name="perch"
loss_type="elbo"
weights_='imagenet'
# layers=[5,2]
initial_lr=0.01
M=3
size=50 #if data_name=="perch" else 20
latent_dim=1024 #15,50, 100, 128, 256,512
beta=1 #0,1, 5, 50,100,200
# if loss_type != "elbo": beta=1
type_="VAE"
if loss_type=="mmd": type_="Info VAE"
if (beta > 1) & (loss_type=="elbo"): type_="Beta VAE"
if beta==0 : type_="Auto-encoder"

weight_path=os.path.join(result_dir,"embeddings/VAE_{}_{}_z{}_b{}_w_{}_weights.ckpt".\
                         format(data_name,loss_type,latent_dim,beta,weights_))
embed_path=os.path.join(result_dir,"embeddings/VAE_{}_{}_z{}_b{}_w_{}_embeddings.pkl".\
                        format(data_name,loss_type,latent_dim,beta,weights_))
dropbox_path=os.path.join(dropbox_dir,"embeddings/data/VAE_{}_{}_z{}_b{}_w_{}_embeddings.pkl".\
                          format(data_name,loss_type,latent_dim,beta,weights_))

ens_weight_path=os.path.join(result_dir,"embeddings/VAE_{}_{}_z{}_b{}_w_{}_weights.pkl".\
                         format(data_name,loss_type,latent_dim,beta,weights_))



batch_size=24
num_examples_to_generate=16

random_vector_for_generation=tf.random.normal(
    shape=[num_examples_to_generate,latent_dim]
)

model=CVAE(latent_dim,input_shape=input_shape,loss=loss_type,layers_decoder=5,l2=l2_,weights=weights_)
# model.build((batch_size,)+input_shape)
if warm_start: model.load_weights(weight_path)

if data_name=="perch":
    path_train,labs_train,path_test,labs_test=perch.load_data()
elif data_name=='chestray':
    path_train_perch, labs_train_perch, path_test_perch, labs_test_perch = perch.load_data()
    path_train_chestray, labs_train_chestray, path_test_chestray, labs_test_chestray = chestray.load_data()
    path_train=np.concatenate([path_train_chestray,
                        np.random.choice(path_train_perch,size=len(path_train_chestray),replace=True)
                              ])
    path_train=np.random.permutation(path_train)
    path_test=np.concatenate([path_test_chestray,path_test_perch])
else:
    raise Exception("Unknown data-set %s" % data_name)

train_im=tf.data.Dataset.from_tensor_slices(path_train)
test_im=tf.data.Dataset.from_tensor_slices(path_test)

load_and_preprocess_image_=partial(load_and_preprocess_image,shape=input_shape)

train_ds=train_im.map(load_and_preprocess_image_,num_parallel_calls=AUTOTUNE)
test_ds=test_im.map(load_and_preprocess_image_,num_parallel_calls=AUTOTUNE)

plot_files=[path_test[i] for i in [0,1,5,7,43]]
plot_ds=tf.data.Dataset.from_tensor_slices(plot_files)
plot_ds=plot_ds.map(load_and_preprocess_image_)



augumentations=[

    flip,
    rotate,
    partial(tf.subtract,y=0.5),
    partial(tf.multiply,y=2.0),


]

augumentations_test=[
    partial(tf.subtract,y=0.5),
    partial(tf.multiply,y=2.0),
]

def undo_center_scale(x):
    x=x/2.0
    x=x+0.5
    return x

for aug in augumentations:
    train_ds=train_ds.map(aug)

for aug_test in augumentations_test:
    test_ds=test_ds.map(aug_test)








train_dataset=train_ds.shuffle(5000).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
test_dataset=test_ds.batch(batch_size)

optimizer=tf.keras.optimizers.SGD(initial_lr,
                                   # clipnorm=1.0,
                                   # decay=0.9,
                                   )
lr_list,lr_index=lr_schedule(initial_lr,M=M,size=size)
weights_list=[]
# checkpoint=Checkpoint(model,patience=10)


elbo_s=[]
display_after= size if data_name == "perch" else 1

train_fun = tf.function(train_one_step)

epochs=M*size
epoch=0
if display is None: pdf_sample=PdfPages("/home/pmwaniki/Dropbox/tmp/vae_%s_%d_b_%d_sample.pdf" % (data_name,latent_dim,beta))
for epoch in range(epoch,epochs):
    print(f'Epoch {epoch}: Setting learning rate to {lr_list[epoch]}')
    tf.keras.backend.set_value(optimizer.lr,lr_list[epoch])

    for train_x in train_dataset: #pass
        grad,grad_norm,loss_perch=train_fun(model,train_x,optimizer,norm=1.0,beta=beta,sigmoid=False,training=True,per_pixel=True)
    loss = tf.keras.metrics.Mean()
    for test_x in test_dataset.take(50):
        loss(compute_loss(model, test_x, beta=beta, sigmoid=sigmoid_loss,training=False))
    elbo = loss.result()
    elbo_s.append([loss_perch.numpy(), elbo.numpy()])
    print("Epoch: {},  loss: {:,.0f}| val_loss: {:,.0f}".format(epoch,loss_perch.numpy(), elbo))

    if epoch in lr_index:
        weights_list.append([w.numpy() for w in model.weights])

    if (epoch % display_after) == 0:
        if display:
            generate_and_save_images(model,random_vector_for_generation,display=True,
                                     transform_fun=undo_center_scale)
        else:
            fig0=generate_and_save_images(model,random_vector_for_generation,display=False,
                                          transform_fun=undo_center_scale)
            pdf_sample.savefig(fig0)

if display is None: pdf_sample.close()




model.save_weights(weight_path)

if data_name == "chestray":
    train_im_perch = tf.data.Dataset.from_tensor_slices(path_train_perch)
    test_im_perch = tf.data.Dataset.from_tensor_slices(path_test_perch)
    train_ds_perch = train_im_perch.map(load_and_preprocess_image_, num_parallel_calls=AUTOTUNE)
    test_ds_perch = test_im_perch.map(load_and_preprocess_image_, num_parallel_calls=AUTOTUNE)
else:
    train_ds_perch=train_ds
    test_ds_perch=test_ds

z_train=np.concatenate([model.embed(images) for images in train_ds_perch.batch(10)])
z_test=np.concatenate([model.embed(images) for images in test_ds_perch.batch(10)])

with open(embed_path ,'wb') as f:
    pickle.dump([z_train,z_test],f)

with open(dropbox_path ,'wb') as f:
    pickle.dump([z_train,z_test],f)

plt.plot(elbo_s)
# plt.ylim(0,2000)
if display: plt.show()


if display:
    plot_reconstruction(model, data=test_ds.shuffle(500), n_images=4, replications=4, shape=input_shape,
                        transform_fun=undo_center_scale)
    plot_reconstruction(model, data=test_ds_perch.shuffle(500), n_images=4, replications=4, shape=input_shape,
                        transform_fun=undo_center_scale)
else:
    fig1=plot_reconstruction(model, data=test_ds.shuffle(500), n_images=4, replications=4, shape=input_shape,
                        transform_fun=undo_center_scale,show=False)
    fig2=plot_reconstruction(model, data=test_ds_perch.shuffle(500), n_images=4, replications=4, shape=input_shape,
                        transform_fun=undo_center_scale,show=False)
    pdf=PdfPages("/home/pmwaniki/Dropbox/tmp/vae_%s_%d_b_%d_reconstruction.pdf" % (data_name,latent_dim,beta))
    pdf.savefig(fig1)
    pdf.savefig(fig2)
    pdf.close()

