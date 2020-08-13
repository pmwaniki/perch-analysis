import tensorflow as tf
import importlib
import os
from settings import result_dir


# def preprocess_resnetv2(x):
#     return x-0.5



models_ = dict(
            VGG16=dict(
                input_shape=(224, 224, 3),
                module_name="vgg16",
                last_conv_layers=["block5_conv3","block5_conv2","block5_conv1"],
                source='keras',
                preprocess_fun=tf.keras.applications.vgg16.preprocess_input,
            ),
            VGG19=dict(
                input_shape=(224, 224, 3),
                module_name="vgg19",
                last_conv_layers=["block5_conv4"],
                source='keras',
                preprocess_fun=tf.keras.applications.vgg19.preprocess_input
            ),
            DenseNet121=dict(
                input_shape=(224, 224, 3),
                module_name="densenet",
                last_conv_layers=["conv5_block16_2_conv","conv5_block16_1_conv",'conv5_block15_2_conv'],
                source='keras',
                preprocess_fun=tf.keras.applications.densenet.preprocess_input
            ),
            ResNet50=dict(
                input_shape=(224, 224, 3),
                module_name="resnet50",
                last_conv_layers=["res5c_branch2c","res5c_branch2b","res5c_branch2a"],
                source='keras',
                preprocess_fun=tf.keras.applications.resnet50.preprocess_input
            ),
            ResNet50V2=dict(
                            input_shape=(224, 224, 3),
                            module_name="resnet50v2",
                            last_conv_layers=["res5c_branch2c","res5c_branch2b","res5c_branch2a"],
                            source='keras',
                            preprocess_fun=tf.keras.applications.resnet_v2.preprocess_input,
                        ),
            ResNet101=dict(
                input_shape=(224, 224, 3),
                module_name="resnet",
                last_conv_layers=["res5c_branch2c", "res5c_branch2b", "res5c_branch2a"],
                source='keras',
                preprocess_fun=tf.keras.applications.resnet.preprocess_input,
            ),
            InceptionV3=dict(
                input_shape=(299, 299, 3),
                module_name="inception_v3",
                last_conv_layers=["mixed10"],
                source='keras',
                preprocess_fun=tf.keras.applications.inception_v3.preprocess_input,
            ),
            InceptionResNetV2=dict(
                input_shape=(299, 299, 3),
                module_name="inception_resnet_v2",
                last_conv_layers=["conv_7b_ac"],
                source='keras',
                preprocess_fun=tf.keras.applications.inception_resnet_v2.preprocess_input,
            ),
            NASNetMobile=dict(
                input_shape=(224, 224, 3),
                module_name="nasnet",
                last_conv_layers=["activation_188"],
                source='keras',
                preprocess_fun=tf.keras.applications.nasnet.preprocess_input,
            ),
            NASNetLarge=dict(
                input_shape=(331, 331, 3),
                module_name="nasnet",
                last_conv_layers=["activation_260"],
                source='keras',
                preprocess_fun=tf.keras.applications.nasnet.preprocess_input,
            )

        )

class BaseModel:
    def __init__(self,name):
        self.models=models_
        self.name=name
        self.input_shape=self.models[name]['input_shape']
        self.module_name=self.models[name]['module_name']


    def __repr__(self):
        return("Model {}| input shape: {}".format(self.name,self.input_shape))

vgg16=BaseModel('VGG16')
resnet50=BaseModel("ResNet50")
resnet50v2=BaseModel("ResNet50V2")
resnet101=BaseModel("ResNet101")
densnet121=BaseModel("DenseNet121")
inceptionv3=BaseModel("InceptionV3")
nasnetmobile=BaseModel("NASNetMobile")
inception_resnet=BaseModel("InceptionResNetV2")
nasnetlarge=BaseModel("NASNetLarge")
vgg19=BaseModel("VGG19")



class Convnet(tf.keras.Model):
    def __init__(self,base_model,n_out=15,multi_label=False,pooling='avg',weights='imagenet',l2=0.0005,
                 ):
        super(Convnet,self).__init__()
        self.base_model=base_model
        self.pooling=pooling
        base_model_class=getattr(
            importlib.import_module(
                "tensorflow.keras.applications"
            ),self.base_model.name
        )
        self.conv_layers=base_model_class(input_shape=base_model.input_shape,
                                     weights=weights,include_top=False,pooling=pooling)
        self.conv_layers_names=[l.name for l in self.conv_layers.layers]
        self.conv_weight_names=[w.name for w in self.conv_layers.weights if w.trainable]
        x=self.conv_layers.output

        self.conv_layers_ = tf.keras.Model(inputs=self.conv_layers.input, outputs=self.conv_layers.output)

        self.output_layer=tf.keras.layers.Dense(n_out,name="logits",activation="sigmoid" if multi_label else "softmax" ,
                                kernel_regularizer=tf.keras.regularizers.l2(l2),
                                kernel_initializer=tf.keras.initializers.he_uniform(),
                                                )

    def call(self,inputs,training=False):
        x=self.conv_layers_(inputs)
        x=self.output_layer(x)
        return x

    def freeze_conv_layers(self):
        for l in self.conv_layers.layers:
            l.trainable = False


    def unfreeze_conv_layers(self,last=None):
        if last is not None:
            n_layers=len(self.conv_layers.layers)
            for l in self.conv_layers.layers[int(last*n_layers):]:
                l.trainable=True
        else:
            for l in self.conv_layers.layers:
                l.trainable=True

    def freeze_dense_layers(self):
        for l in self.model.layers:
            if l.name in ['logits','dense1']:
                l.trainable=False

    def unfreeze_dense_layers(self):
        for l in self.model.layers:
            if l.name in ['logits','dense1']:
                l.trainable=True

    def get_embedding(self,input):
        pass






class ChestnetModel(Convnet):
    def __init__(self,base_model,n_out=15,pooling='avg',weights='imagenet',l2=0.0005,
                 ):
        super(ChestnetModel,self).__init__(base_model=base_model,n_out=n_out,multi_label=True,pooling=pooling,weights=weights,l2=l2,
                                           )
        self.pooling=pooling
        self.filepath_conv = os.path.join(result_dir,
                                     "finetune/chestnet_{0}_{1}_w_{2}_convlayers.ckpt".format(self.base_model.name,
                                                                                        self.pooling,weights))
        self.filepath_net = os.path.join(result_dir,
                                    "finetune/chestnet_{0}_{1}_w_{2}.ckpt".format(self.base_model.name,
                                                                            self.pooling,weights))

    def save_weights(self, filepath=None, overwrite=True, save_format=None):
        if filepath is None:
            self.conv_layers.save_weights(self.filepath_conv,overwrite=overwrite,save_format=save_format)
            super(ChestnetModel,self).save_weights(self.filepath_net,overwrite=overwrite,save_format=save_format)
        else:
            super(ChestnetModel,self).save_weights(filepath,overwrite=overwrite,save_format=save_format)

    def load_weights(self, filepath=None, by_name=False):
        if filepath is None:
            super(ChestnetModel,self).load_weights(self.filepath_net,by_name=by_name)
        else:
            super(ChestnetModel, self).load_weights(filepath, by_name=by_name)




class PerchModel(Convnet):
    def __init__(self,base_model,n_out=5,pooling='avg',weights='imagenet',weight_path=None,multilabel=False,l2=0.0005,
                 init_weights='imagenet'):
        if weights not in ["chestnet", "imagenet", None]:
            raise Exception("Unknow weight initialization")
        self.weights_='imagenet' if weights =='imagenet' else None
        super(PerchModel,self).__init__(base_model=base_model,n_out=n_out,multi_label=multilabel,pooling=pooling,weights=self.weights_,l2=l2,
                                        )
        self.pooling = pooling

        self.filepath_net = os.path.join(result_dir,
                                         "finetune/perch_{0}_{1}_{2}.ckpt".format(self.base_model.name,self.pooling, weights))
        if weight_path is None:
            weight_path = os.path.join(result_dir,
                                       "finetune/chestnet_{0}_{1}_w_{2}_convlayers.ckpt".format(self.base_model.name,
                                                                                          self.pooling,init_weights))
        if weights == 'chestnet':
            print("loading chestnet weights")
            self.conv_layers_.load_weights(weight_path)

    def save_weights(self, filepath=None, overwrite=True, save_format=None):
        if filepath is None:

            super(PerchModel,self).save_weights(self.filepath_net, overwrite=overwrite,
                                    save_format=save_format)

        else:

            super(PerchModel, self).save_weights(filepath, overwrite=overwrite,
                                    save_format=save_format)


    def load_weights(self, filepath=None, by_name=False):
        if filepath is None:
            super(PerchModel,self).load_weights(self.filepath_net, by_name=by_name)
        else:
            super(PerchModel, self).load_weights(filepath, by_name=by_name)


self=PerchModel(base_model=vgg16)


class Multitask(tf.keras.Model):
    def __init__(self,base_model,n_out=[5,15],pooling='avg',weights='imagenet',l2=0.0005,
                 ):
        super(Multitask,self).__init__()
        self.base_model=base_model
        self.file_path_perch=os.path.join(result_dir,
                                    "multitask/perch_{0}_{1}_l2_{2}.ckpt".format(base_model.name,
                                                                            pooling,l2,))
        self.file_path_chestray=os.path.join(result_dir,
                                    "multitask/chestray_{0}_{1}_l2_{2}.ckpt".format(base_model.name,
                                                                            pooling,l2,))
        base_model_class=getattr(
            importlib.import_module(
                "tensorflow.keras.applications"
            ),self.base_model.name
        )
        self.conv_layers=base_model_class(input_shape=base_model.input_shape,
                                     weights=weights,include_top=False,pooling=pooling)
        self.conv_layers_names=[l.name for l in self.conv_layers.layers]
        self.conv_weight_names=[w.name for w in self.conv_layers.weights if w.trainable]
        x=self.conv_layers.output
        if pooling is None:
            x=tf.keras.layers.Flatten()(x)


        out_perch=tf.keras.layers.Dense(n_out[0],name="logits_perch",activation= "softmax",
                                kernel_regularizer=tf.keras.regularizers.l2(l2),
                                        kernel_initializer=tf.keras.initializers.he_uniform(),
                                        )(x)
        out_chestray = tf.keras.layers.Dense(n_out[1], name="logits_chestray", activation="sigmoid",
                                          kernel_regularizer=tf.keras.regularizers.l2(l2),
                                             kernel_initializer=tf.keras.initializers.he_uniform(),
                                             )(x)
        self.model_perch=tf.keras.models.Model(inputs=self.conv_layers.input,outputs=out_perch)
        self.model_chestray=tf.keras.models.Model(inputs=self.conv_layers.input,outputs=out_chestray)
        self.perch_layers=set(self.model_perch.layers)
        self.chestray_layers=set(self.model_chestray.layers)
        self.all_layers=self.perch_layers.union(self.chestray_layers)
        self.convolution_layers=set(self.conv_layers.layers)
        self.dense_layers=self.all_layers.difference(self.convolution_layers)

    def call(self,inputs):
        perch,chestray=inputs
        if perch is not None:
            perch_y=self.model_perch(perch)
        else:
            perch_y=None
        if chestray is not None:
            chestray_y=self.model_chestray(chestray)
        else:
            chestray_y=None

        return [perch_y,chestray_y]

    def save_weights(self, filepath=None, overwrite=True, save_format=None):
        if filepath is None:
            self.model_perch.save_weights(self.file_path_perch,overwrite=overwrite,save_format=save_format)
            self.model_chestray.save_weights(self.file_path_chestray,overwrite=overwrite,save_format=save_format)
        else:
            self.model_perch.save_weights(os.path.join(filepath,"perch"), overwrite=overwrite, save_format=save_format)
            self.model_chestray.save_weights(os.path.join(filepath,"chestray"), overwrite=overwrite,
                                                   save_format=save_format)

    def load_weights(self, filepath=None, by_name=False):
        if filepath is None:
            self.model_perch.load_weights(self.file_path_perch,by_name=by_name)
            self.model_chestray.load_weights(self.file_path_chestray,by_name=by_name)
        else:
            self.model_perch.load_weights(os.path.join(filepath,"perch"), by_name=by_name)
            self.model_chestray.load_weights(os.path.join(filepath,"chestray"), by_name=by_name)

    def freeze_conv_layers(self):
        for l in self.conv_layers.layers:
            l.trainable = False


    def unfreeze_conv_layers(self,last=None):
        if last is not None:
            n_layers=len(self.conv_layers.layers)
            for l in self.conv_layers.layers[int(last*n_layers):]:
                l.trainable=True
        else:
            for l in self.conv_layers.layers:
                l.trainable=True

    def freeze_dense_layers(self):
        for l in self.dense_layers:
            l.trainable=False

    def unfreeze_dense_layers(self):
        for l in self.dense_layers:
            l.trainable = True



class ConditioningLayer(tf.keras.layers.Layer):
    def __init__(self,conditioning="multiply",l2=None,activation=None,**kwargs):
        super(ConditioningLayer,self).__init__(**kwargs)
        self.conditioning=conditioning
        self.l2=l2
        if conditioning=="multiply":
            self.combine=tf.keras.layers.Multiply()
        elif conditioning=="add":
            self.combine=tf.keras.layers.Add()
        else:
            raise Exception("Unknown conditioning operator: %s" % conditioning)
        self.sigmoid=tf.keras.layers.Activation(activation)

    def build(self, input_shape):
        assert isinstance(input_shape,list)
        image_embed_shape, rev_embed_shape = input_shape
        self.kernel=self.add_weight(name="linear_kernel",shape=(rev_embed_shape[1],image_embed_shape[1]),
                                    initializer=tf.keras.initializers.he_uniform(),
                                    trainable=True)
        super(ConditioningLayer,self).build(input_shape)
        pass

    def call(self,inputs):
        image_embed,rev_embed=inputs
        x=tf.matmul(rev_embed,self.kernel)
        x=self.combine([x,image_embed])
        if self.l2:
            self.add_loss(self.l2*tf.reduce_sum(tf.square(self.kernel)))
        x=self.sigmoid(x)
        return x

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        image_embed_shape, rev_embed_shape = input_shape
        return image_embed_shape


class MultitaskEnsemble(tf.keras.Model):
    def __init__(self,base_model,n_out=[5,15],n_raters=18,n_embedding=16,pooling='avg',
                 weights='imagenet',l2=0.0005,l2_embeddings=0.0005,
                 conditioning="multiply",activation_conditioning="sigmoid"):
        super(MultitaskEnsemble,self).__init__()
        self.base_model=base_model
        self.file_path_perch=os.path.join(result_dir,
                                    "multitask/perch_ensemble_{0}_{1}_l2_{2}.ckpt".format(base_model.name,
                                                                            pooling,l2))
        self.file_path_chestray=os.path.join(result_dir,
                                    "multitask/chestray_ensemble_{0}_{1}_l2_{2}.ckpt".format(base_model.name,
                                                                            pooling,l2))
        base_model_class=getattr(
            importlib.import_module(
                "tensorflow.keras.applications"
            ),self.base_model.name
        )
        self.conv_layers=base_model_class(input_shape=base_model.input_shape,
                                     weights=weights,include_top=False,pooling=pooling)
        self.conv_layers_names=[l.name for l in self.conv_layers.layers]
        self.conv_weight_names=[w.name for w in self.conv_layers.weights if w.trainable]
        x=self.conv_layers.output
        if pooling is None:
            x=tf.keras.layers.Flatten()(x)


        embedding_input=tf.keras.layers.Input(shape=())
        embedding_layer = tf.keras.layers.Embedding(input_dim=n_raters, output_dim=n_embedding,
                                                         input_length=None,
                                                         embeddings_initializer=tf.keras.initializers.he_uniform(),
                                                         embeddings_regularizer=tf.keras.regularizers.l2(l2_embeddings),
                                                         )(embedding_input)
        embedding_layer = tf.keras.layers.Flatten()(embedding_layer)
        x2=ConditioningLayer(conditioning=conditioning,l2=l2_embeddings,name="Conditioning_Layer",
                                                    activation=activation_conditioning)([x,embedding_layer])



        out_perch=tf.keras.layers.Dense(n_out[0],name="logits_perch",activation= "softmax",
                                kernel_regularizer=tf.keras.regularizers.l2(l2),
                                        kernel_initializer=tf.keras.initializers.he_uniform())(x2)
        out_chestray = tf.keras.layers.Dense(n_out[1], name="logits_chestray", activation="sigmoid",
                                          kernel_regularizer=tf.keras.regularizers.l2(l2),
                                             kernel_initializer=tf.keras.initializers.he_uniform())(x)
        self.model_perch=tf.keras.models.Model(inputs=[self.conv_layers.input,embedding_input],outputs=out_perch)
        self.model_chestray=tf.keras.models.Model(inputs=self.conv_layers.input,outputs=out_chestray)
        self.im_embedding_model=tf.keras.models.Model(inputs=self.conv_layers.input,outputs=x)
        self.perch_layers=set(self.model_perch.layers)
        self.chestray_layers=set(self.model_chestray.layers)
        self.all_layers=self.perch_layers.union(self.chestray_layers)
        self.convolution_layers=set(self.conv_layers.layers)
        self.dense_layers=self.all_layers.difference(self.convolution_layers)

    def call(self,inputs):
        perch,chestray=inputs
        if perch is not None:
            perch_y=self.model_perch(perch)
        else:
            perch_y=None
        if chestray is not None:
            chestray_y=self.model_chestray(chestray)
        else:
            chestray_y=None

        return [perch_y,chestray_y]

    def embed_images(self,images):
        return self.im_embedding_model(images)

    def save_weights(self, filepath=None, overwrite=True, save_format=None):
        if filepath is None:
            self.model_perch.save_weights(self.file_path_perch,overwrite=overwrite,save_format=save_format)
            self.model_chestray.save_weights(self.file_path_chestray,overwrite=overwrite,save_format=save_format)
        else:
            self.model_perch.save_weights(os.path.join(filepath,"perch"), overwrite=overwrite, save_format=save_format)
            self.model_chestray.save_weights(os.path.join(filepath,"chestray"), overwrite=overwrite,
                                                   save_format=save_format)

    def load_weights(self, filepath=None, by_name=False):
        if filepath is None:
            self.model_perch.load_weights(self.file_path_perch,by_name=by_name)
            self.model_chestray.load_weights(self.file_path_chestray,by_name=by_name)
        else:
            self.model_perch.load_weights(os.path.join(filepath,"perch"), by_name=by_name)
            self.model_chestray.load_weights(os.path.join(filepath,"chestray"), by_name=by_name)

    def freeze_conv_layers(self):
        for l in self.conv_layers.layers:
            l.trainable = False


    def unfreeze_conv_layers(self,last=None):
        if last is not None:
            n_layers=len(self.conv_layers.layers)
            for l in self.conv_layers.layers[int(last*n_layers):]:
                l.trainable=True
        else:
            for l in self.conv_layers.layers:
                l.trainable=True

    def freeze_dense_layers(self):
        for l in self.dense_layers:
            l.trainable=False

    def unfreeze_dense_layers(self):
        for l in self.dense_layers:
            l.trainable = True








class PerchEnsembleModel(tf.keras.Model):
    def __init__(self,base_model,n_out=5,n_raters=18,n_embedding=16,
                 pooling='avg',weights=None,init_weights='imagenet',
                 weight_path=None,multilabel=False,l2=0.0005,
                 l2_embeddings=0.005,
                 conditioning="add",activation_conditioning="sigmoid"):
        super(PerchEnsembleModel,self).__init__()
        convnet=PerchModel(base_model,n_out=n_out,pooling=pooling,
                        weights=weights,weight_path=weight_path,l2=l2,
                           init_weights=init_weights)
        self.conv_layers=tf.keras.Model(inputs=convnet.conv_layers.input,outputs=convnet.conv_layers.output)
        self.embedding_layer = tf.keras.layers.Embedding(input_dim=n_raters, output_dim=n_embedding,
                                                         input_length=None,
                                                         embeddings_initializer=tf.keras.initializers.he_uniform(),
                                                         # embeddings_initializer=tf.keras.initializers.RandomNormal(
                                                         #     stddev=0.01),
                                                         embeddings_regularizer=tf.keras.regularizers.l2(l2_embeddings),
                                                         )
        self.reshape_layer=tf.keras.layers.Flatten()
        self.conditioning_layer = ConditioningLayer(conditioning=conditioning,l2=l2_embeddings,name="Conditioning_Layer",
                                                    activation=activation_conditioning)

        self.logits = tf.keras.layers.Dense(n_out, activation='softmax' if not multilabel else "sigmoid",
                                            kernel_regularizer=tf.keras.regularizers.l2(l2),
                                            kernel_initializer=tf.keras.initializers.he_uniform(),
                                            name="logits")

        self.filepath_net = os.path.join(result_dir,
                                         "finetune/perch_ensemble_{3}{0}_{1}_{2}.ckpt".format(base_model.name,
                                                                                           pooling, weights,
                                                                                              "" if not multilabel else "multilabel_"))

    def call(self,inputs,training=False):
        im,emb=inputs
        x1=self.conv_layers(im)
        x2=self.embedding_layer(emb)
        x2=self.reshape_layer(x2)
        x=self.conditioning_layer([x1,x2])
        x=self.logits(x)
        return x

    def embed_images(self,images):
        return self.conv_layers(images)

    def embed_reviewers(self,reviwers):
        return self.embedding_layer(reviwers)


    def save_weights(self, filepath=None, overwrite=True, save_format=None):
        if filepath is None:
            super(PerchEnsembleModel,self).save_weights( self.filepath_net, overwrite=overwrite,
                                    save_format=save_format)

        else:
            super(PerchEnsembleModel,self).save_weights(filepath, overwrite=overwrite,
                                    save_format=save_format)


    def load_weights(self, filepath=None, by_name=False):
        if filepath is None:
            super(PerchEnsembleModel,self).load_weights( self.filepath_net, by_name=by_name)
        else:
            super(PerchEnsembleModel,self).load_weights(filepath, by_name=by_name)

    def freeze_conv_layers(self):
        for l in self.conv_layers.layers:
            l.trainable = False


    def unfreeze_conv_layers(self,last=None):
        if last is not None:
            n_layers=len(self.conv_layers.layers)
            for l in self.conv_layers.layers[int(last*n_layers):]:
                l.trainable=True
        else:
            for l in self.conv_layers.layers:
                l.trainable=True

    def freeze_dense_layers(self):
        for l in self.layers:
            if l.name in ['logits','dense1','dense2','dense3','dense4']:
                l.trainable=False

    def unfreeze_dense_layers(self):
        for l in self.layers:
            if l.name in ['logits','dense1','dense2','dense3','dense4']:
                l.trainable=True




def rev_weights_model(input_shape,dropout=0.2,l2=0.005,gaus_noise=0.05):
    return tf.keras.models.Sequential([
    tf.keras.layers.GaussianNoise(gaus_noise,input_shape=input_shape),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(512,activation=None,
                          kernel_initializer=tf.keras.initializers.GlorotNormal(),
                          kernel_regularizer=tf.keras.regularizers.l2(l2),
                          use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(521,activation=None,
                          kernel_regularizer=tf.keras.regularizers.l2(l2),
                          kernel_initializer=tf.keras.initializers.GlorotNormal(),use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(100,activation=None,
                          kernel_regularizer=tf.keras.regularizers.l2(l2),
                          kernel_initializer=tf.keras.initializers.GlorotNormal(),use_bias=True),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(1,activation="sigmoid",kernel_regularizer=tf.keras.regularizers.l2(l2),
                          kernel_initializer=tf.keras.initializers.GlorotNormal(),use_bias=True)
])