import tensorflow as tf
from tensorflow.keras.layers import Conv2D , Add , Dense
def discriminator_block_1(X): 
    X_short = X

    X = Conv2D(filters = 32, kernel_size = (3,3) , strides = (1,1), padding = 'same')(X)
    X = tf.keras.layers.LeakyReLU()(X)
    X = Conv2D(filters = 32, kernel_size = (3,3) , strides = (1,1), padding = 'same')(X)
    X = Add()([X , X_short])

    X = tf.keras.layers.LeakyReLU()(X)
    return X
def discriminator_block_2(X, filters): 
    X_short = X

    X = Conv2D(filters = filters, kernel_size = (3,3) , strides = (1,1), padding = 'same')(X)
    X = tf.keras.layers.LeakyReLU()(X)
    X = Conv2D(filters = filters, kernel_size = (3,3) , strides = (1,1), padding = 'same')(X)
    X = tf.keras.layers.LeakyReLU()(X)
    X = Add()([X , X_short])

    X = tf.keras.layers.LeakyReLU()(X)
    return X
def discriminator_block_3(X, filters, kernel): 
    X = Conv2D(filters = filters, kernel_size = (kernel,kernel) , strides = (2,2), padding = 'same')(X)
    X = tf.keras.layers.LeakyReLU()(X)

    for i in range(2):
        X = discriminator_block_2(X , filters)
    
    return X

def discriminator_model(input_shape = (128 , 128 , 3)):
    X_input = tf.keras.Input(input_shape)

    X = Conv2D(filters = 32, kernel_size = (4,4) , strides = (2,2))(X_input)
    X = tf.keras.layers.LeakyReLU()(X)

    for i in range(2):
         X = discriminator_block_1(X)
    
    X = Conv2D(filters = 64, kernel_size = (4,4) , strides = (2,2))(X)
    X = tf.keras.layers.LeakyReLU()(X)

    X = discriminator_block_3(X, 64, 4)
    X = discriminator_block_3(X, 128, 4)
    X = discriminator_block_3(X, 256, 3)
    X = discriminator_block_3(X, 512, 3)

    X = Conv2D(filters = 1024, kernel_size = (3,3) , strides = (2,2), padding = 'same')(X)
    X = tf.keras.layers.LeakyReLU()(X)


    X = tf.keras.layers.Flatten()(X)


    discrimination = Dense(1)(X)
    discrimination = tf.keras.activations.sigmoid(discrimination)

    tags = tf.keras.layers.Dense(34 )(X)
    tags = tf.keras.activations.sigmoid(tags)

    Model = tf.keras.Model(inputs = X_input , outputs = [discrimination , tags])

    return Model 

def discriminator_loss(real , fake, real_tags): 
    
    real_genuity , real_attributes = real[0] , real[1]
    fake_genuity , fake_attributes = fake[0] , fake[1]


    negative_one = tf.constant(-1)

    L_adv = K.update_add(
                        K.log(real_genuity) , 
                        K.log(K.update_sub(K.ones(fake_genuity.shape), fake_genuity))
                        )
    L_adv = K.sum( L_adv ,axis = 1, keepdims = True)
    L_adv = tf.math.scalar_mul(L_adv , negative_one)
    L_adv = K.sum( L_adv ,axis = 1, keepdims = True)

    # Since I could not understand L_cls completely while reading the paper, it had to improvise
    L_cls = K.sum(
            K.update_add(
                        K.log(K.abs(K.update_sub (real_tags, real_attributes) )),
                        K.log(K.abs(K.update_sub(real_tags, fake_attributes)))
                        )
                    , axis = 1 , keepdims =True)

    return K.sum(L.csl , tf.scalar_mul(L_adv , tf.constant(34)))