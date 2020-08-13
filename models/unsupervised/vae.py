import tensorflow as tf
import numpy as np
from models.unsupervised.autoencoders import dense_encoder,dense_decoder
from models.supervised.applications import ConditioningLayer

import matplotlib.pyplot as plt

input_shape=(224,224,3)
latent_dim=15
class CVAE(tf.keras.Model):
    def __init__(self,latent_dim,input_shape,loss="elbo",layers_decoder=5,l2=0.0005,weights=None):
        super(CVAE,self).__init__()
        self.latent_dim=latent_dim
        self.loss=loss
        self.inference_net=dense_encoder(input_shape,
                                   latent_dim=(latent_dim+latent_dim) if loss=="elbo" else latent_dim,
                                   l2=l2,weights=weights)
        self.generative_net=dense_decoder(input_shape,latent_dim=latent_dim,layers=layers_decoder,l2=l2,activation='tanh')



    def call(self,inputs,training=False):
        z=self.embed(inputs,training=training)
        return self.decode(z,training=training)

    def sample(self,eps=None):
        if eps is None:
            eps=tf.random.normal(shape=(100,self.latent_dim))
        return self.decode(eps)

    def encode(self,x,training=False):
        if self.loss=="elbo":
            mean,logvar=tf.split(self.inference_net(x,training=training),num_or_size_splits=2,axis=1)
            return mean,logvar
        else:
            return self.inference_net(x,training=training)

    def reparameterize(self,mean,logvar):
        eps=tf.random.normal(shape=mean.shape)
        return eps*tf.exp(logvar * .5) + mean

    def embed(self,x,training=False):

        if self.loss=="elbo":
            mean, logvar = self.encode(x, training=training)
            return self.reparameterize(mean,logvar)
        else:
            return self.encode(x,training=training)

    def reconstruct(self,x):
        z=self.embed(x,training=False)
        return self.decode(z,training=False)

    def decode(self,z,training=False):
        logits=self.generative_net(z,training=training)
        return logits


class Classifier(tf.keras.Model):
    def __init__(self,model,n_out=5,l2=0.0005,
                 # dropout=0.05
                 ):
        super(Classifier,self).__init__()
        self.loss=model.loss
        self.inference_net_=model.inference_net
        inf_net_input=model.inference_net.input
        inf_net_output=model.inference_net.get_layer("ConvnetGlobalAveragePooling").output



        self.inference_net=tf.keras.Model(inputs=inf_net_input,outputs=inf_net_output)
        self.DenseLayer=tf.keras.layers.Dense(n_out,name="predictions",activation= "softmax" ,
                                kernel_regularizer=tf.keras.regularizers.l2(l2),
                                kernel_initializer=tf.keras.initializers.he_uniform(),
                                              )
    def call(self,inputs, *args, **kwargs):
        x=self.inference_net(inputs)
        x=self.DenseLayer(x)
        return x

    def split_sample(self,inputs):
        mean, logvar = tf.split(inputs, num_or_size_splits=2, axis=1)
        eps = tf.random.normal(tf.shape(mean))
        x = eps * tf.exp(logvar * .5) + mean
        return x
    def freeze_conv_layers(self):
        for l in self.inference_net.layers:
            l.trainable=False
    def unfreeze_conv_layers(self,last=None):
        for l in self.inference_net.layers:
            l.trainable=True


class ClassifierEnsemble(tf.keras.Model):
    def __init__(self,model,n_out=5,n_raters=18,n_embedding=16,l2=0.0005,l2_embeddings=0.0005,
                 # dropout=0.05,
                 conditioning="multiply",
                 activation_conditioning="sigmoid"

                 ):
        super(ClassifierEnsemble,self).__init__()
        self.loss=model.loss
        self.inference_net_=model.inference_net
        inf_net_input = model.inference_net.input
        inf_net_output = model.inference_net.get_layer("ConvnetGlobalAveragePooling").output




        self.inference_net=tf.keras.Model(inputs=inf_net_input,outputs=inf_net_output)

        #Embeddings

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=n_raters, output_dim=n_embedding,
                                                         input_length=None,
                                                         embeddings_initializer=tf.keras.initializers.he_uniform(),
                                                         # embeddings_initializer=tf.keras.initializers.RandomNormal(
                                                         #     stddev=0.01),
                                                         embeddings_regularizer=tf.keras.regularizers.l2(l2_embeddings),
                                                         )
        self.reshape_layer = tf.keras.layers.Flatten()
        self.conditioning_layer=ConditioningLayer(conditioning=conditioning,l2=l2_embeddings,name="Conditioning_Layer",
                                                    activation=activation_conditioning)





        self.logits = tf.keras.layers.Dense(n_out, activation='softmax',
                                            kernel_regularizer=tf.keras.regularizers.l2(l2),
                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                            name="logits")



    def call(self,inputs, *args, **kwargs):
        im, emb = inputs
        x1 = self.inference_net(im)
        x2 = self.embedding_layer(emb)
        x2 = self.reshape_layer(x2)
        x=self.conditioning_layer([x1,x2])
        x = self.logits(x)
        return x

    def embed_images(self,images):
        return self.inference_net(images)

    def embed_reviewer(self,rev):
        return self.embedding_layer(rev)




class PerceptionLoss:
    def __init__(self):
        model=tf.keras.applications.vgg16.VGG16(input_shape=(224,224,3),weights='imagenet',include_top=False)
        model.trainable=False
        for layer in model.layers:
            layer.trainable=False

        self.feature_model=tf.keras.Model(inputs=model.input,outputs=model.get_layer("block1_pool").output)
    def extract_features(self,x):
        return self.feature_model(x)
    def loss(self,x,y):
        f_x=self.extract_features(x)
        f_y=self.extract_features(y)
        diff=tf.square(f_x-f_y)
        diff2=tf.reduce_sum(diff,axis=[1,2,3])
        l=tf.sqrt(tf.reduce_mean(diff2))
        return l

perceptual_loss=PerceptionLoss()

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)


def compute_kl(mean,logvar):
    return 0.5*tf.reduce_sum(
        tf.exp(logvar)+tf.square(mean)-1-logvar
    )


def log_normal_pdf(sample,mean,logvar,raxis=1):
    log2pi=tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5*((sample-mean)**2 * tf.exp(-logvar)+ logvar + log2pi),
        axis=raxis
    )

def compute_loss(model,x,beta=1,sigmoid=False,training=False,per_pixel=True):
    if model.loss=="elbo":
        mean,logvar=model.encode(x,training=training)
        z=model.reparameterize(mean,logvar)
    else:
        z=model.embed(x,training=training)
    x_hat=model.decode(z,training=training)

    if sigmoid:
        x_binary = tf.cast(x > 0.5, tf.float32)
        recon_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=x_binary)
        logpx_z = tf.reduce_sum(recon_loss, axis=[1, 2, 3]) + tf.add_n(model.losses)
    else:
        recon_loss = tf.losses.mean_squared_error(x, x_hat)
        logpx_z = tf.reduce_sum(recon_loss, axis=[1, 2, ]) + tf.add_n(model.losses)

    if not per_pixel:
        perc_loss=perceptual_loss.loss(x,x_hat)
        logpx_z=logpx_z+perc_loss




    if model.loss == "elbo":
        loss=tf.reduce_mean(logpx_z)+beta*compute_kl(mean,logvar)
    elif model.loss=="mmd":
        true_samples=tf.random.normal(z.shape)
        mmd_loss=compute_mmd(true_samples,z)
        loss=tf.reduce_mean(logpx_z)+beta*mmd_loss
    return loss




def compute_gradients(model,x,beta=1,sigmoid=False,training=True,per_pixel=True):
    with tf.GradientTape() as tape:
        loss=compute_loss(model,x,beta=beta,sigmoid=sigmoid,training=training,per_pixel=per_pixel)
    return tape.gradient(loss,model.trainable_variables),loss

def apply_gradients(optimizer,gradients,variables):
    optimizer.apply_gradients(zip(gradients,variables))


def train_one_step(model,x,optimizer,norm,beta=1,sigmoid=False,training=True,per_pixel=True):
    grad,loss=compute_gradients(model,x,beta=beta,sigmoid=sigmoid,training=training,per_pixel=per_pixel)
    grad_norm=[tf.clip_by_norm(t,norm) for t in grad]
    apply_gradients(optimizer,grad_norm,model.trainable_variables)
    return grad,grad_norm,loss



def generate_and_save_images(model,test_input,display=True,filename=None,transform_fun=None):
    predictions=model.sample(test_input)
    if transform_fun is not None:
        predictions=transform_fun(predictions)
    fig=plt.figure(figsize=(7,7))

    for i in range(predictions.shape[0]):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i,:,:,0],cmap="gray")
        plt.axis("off")

    if filename is not None: plt.savefig(filename)
    if display: plt.show()
    return fig



def plot_reconstruction(model,data,n_images=5,replications=6,shape=(224,224,1),transform_fun=None,show=True):
    h,w,_=shape
    output=np.zeros((n_images*h,(replications+1)*w))
    for i,image in enumerate(data.take(n_images).batch(1)):
        output[i*h:(i+1)*h,0:w]=image.numpy()[0,:,:,0]
        for j in range(1,replications+1):
            dec_image = model.reconstruct(image)
            output[i*h:(i+1)*h,j*w:(j+1)*w]=dec_image[0,:,:,0]

    fig=plt.figure(figsize=(12,12))
    if transform_fun: output=transform_fun(output)
    plt.imshow(output,cmap="gray",vmin=0,vmax=1)
    plt.axis("off")
    if show: plt.show()
    return fig

#
