import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0],True)
layers=tf.keras.layers





def conv_block(x,filters,kernel_size=3,strides=(1,1),activation="relu",padding='same',l2=None):
    h=tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, activation=None,
                           padding=padding, kernel_initializer=tf.keras.initializers.glorot_normal(),
                           kernel_regularizer=l2 if l2 is None else tf.keras.regularizers.l2(l2)
                           )(x)
    h=tf.keras.layers.BatchNormalization()(h)
    h=tf.keras.layers.Activation(activation)(h)
    return h

def dec_block(x,filters,kernel_size=3,l2=0.00005):
    x=tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=(2, 2), padding="same",
                                    activation=None, use_bias=False,
                                    kernel_initializer=tf.keras.initializers.glorot_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l2(l2)
                                    )(x)
    x=tf.keras.layers.BatchNormalization()(x)
    x=tf.keras.layers.Activation("relu")(x)
    return x



def encoder(input_shape=(224,224,1),n_layers=6,latent_dim=10,final_activation=None,l2=None):
    input=tf.keras.Input(input_shape)
    x=input
    for filter_size in [2**i for i in range(4,4+n_layers)]:
        x=conv_block(x,filters=filter_size,l2=l2)
        x=conv_block(x,filters=filter_size,l2=l2)
        x=tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='valid')(x)
    #dense
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(latent_dim,activation=final_activation,
                            kernel_regularizer=l2 if l2 is None else tf.keras.regularizers.l2(l2))(x)
    return tf.keras.Model(inputs=input,outputs=x)

enc=encoder(n_layers=4)

def decoder(input_shape=(224,224,1),latent_dim=100,layers=4,l2=0.000005,activation='tanh'):
    input=tf.keras.layers.Input(shape=(latent_dim,))
    units_reshape=2**(4+layers)
    width=input_shape[0]
    for i in range(layers):width=int(width/2)
    x=tf.keras.layers.Dense(units=width * width * units_reshape, activation=tf.nn.relu)(input)
    x=tf.keras.layers.Reshape(target_shape=(width, width, units_reshape))(x)
    for l in [2**i for i in range(4,4+layers)][::-1]:
        x=dec_block(x,filters=l,kernel_size=3,l2=l2)
    x=tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="same",
                                    activation=activation,
                                    use_bias=False,
                                    kernel_initializer=tf.keras.initializers.glorot_normal(),
                                    kernel_regularizer=tf.keras.regularizers.l2(l2)
                                    )(x)
    return tf.keras.Model(inputs=input,outputs=x)


def discriminator(input_shape=(224,224,1),n_layers=6,n_classes=2,l2=None):
    input=tf.keras.Input(input_shape)
    x=input
    for filter_size in [2**i for i in range(4,4+n_layers)]:
        x=conv_block(x,filters=filter_size,l2=l2)
        x=conv_block(x,filters=filter_size,l2=l2)
        x=tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='valid')(x)
    #dense
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(n_classes,activation='softmax',
                            kernel_regularizer=l2 if l2 is None else tf.keras.regularizers.l2(l2))(x)
    return tf.keras.Model(inputs=input,outputs=x)

def dense_encoder(input_shape,latent_dim,weights=None,l2=0.0005):
    base_model=tf.keras.applications.ResNet50V2(input_shape=input_shape,
                                     weights=weights,include_top=False,pooling=None)
    x=base_model.output
    x=tf.keras.layers.GlobalAveragePooling2D(name="ConvnetGlobalAveragePooling")(x)
    x=tf.keras.layers.Dense(latent_dim,activation=None,
                            kernel_regularizer=l2 if l2 is None else tf.keras.regularizers.l2(l2)
                            )(x)
    return tf.keras.Model(inputs=base_model.inputs,outputs=x)


def dense_decoder(input_shape=(224,224,3),latent_dim=100,layers=4,l2=0.000005,activation='tanh'):
    input = tf.keras.layers.Input(shape=(latent_dim,))
    units_reshape = 2 ** (4 + layers)
    width = input_shape[0]
    for i in range(layers): width = int(width / 2)
    x = tf.keras.layers.Dense(units=width * width * units_reshape, activation=tf.nn.relu)(input)
    x = tf.keras.layers.Reshape(target_shape=(width, width, units_reshape))(x)
    for l in [2 ** i for i in range(4, 4 + layers)][::-1]:
        x = dec_block(x, filters=l, kernel_size=3, l2=l2)
    x = tf.keras.layers.Conv2DTranspose(filters=input_shape[2], kernel_size=3, strides=(1, 1), padding="same",
                                        activation=activation
                                        , use_bias=False,
                                        kernel_initializer=tf.keras.initializers.glorot_normal(),
                                        kernel_regularizer=tf.keras.regularizers.l2(l2)
                                        )(x)
    return tf.keras.Model(inputs=input, outputs=x)

