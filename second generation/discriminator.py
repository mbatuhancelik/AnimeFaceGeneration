import tensorflow as tf
import tensorflow.keras.backend as K
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

def discriminator_model(input_shape = (128 , 128 , 3), num_features = 39):
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

    tags = tf.keras.layers.Dense(num_features)(X)
    tags = tf.keras.activations.sigmoid(tags)

    Model = tf.keras.Model(inputs = X_input , outputs = [discrimination , tags])

    return Model 

def discriminator_loss(real , fake, real_tags, lambda_adv): 
    
    real_genuity , real_attributes = real[0] , real[1]
    fake_genuity , fake_attributes = fake[0] , fake[1]


    negative_one = tf.constant(-1 , dtype = tf.float32 , shape = (1,1))
    lambda_adv = tf.constant(lambda_adv , dtype = tf.float32 , shape = (1,1))

    L_adv = tf.math.add(
                        tf.math.log(real_genuity) , 
                        tf.math.log(tf.math.subtract(tf.ones(fake_genuity.shape), fake_genuity))
                        )
    L_adv = tf.math.multiply(negative_one,L_adv)
    
    #Sadly, I lack the mathematical background to understand how L_cls in the paper works. 
    #This is just the distance between feature vectors
    geniune_real = tf.math.squared_difference(real_tags ,real_attributes)
    geniune_real = tf.math.reduce_sum(geniune_real , axis = 1 , keepdims = True)
    geniune_real = tf.sqrt(geniune_real)
    
    geniune_fake = tf.math.squared_difference(real_tags ,fake_attributes)
    geniune_fake = tf.math.reduce_sum(geniune_fake , axis = 1 , keepdims = True)
    geniune_fake = tf.sqrt(geniune_fake)
    
    L_cls = tf.math.add(geniune_fake , geniune_real)
    
    return tf.math.add(L_cls , tf.multiply(L_adv , lambda_adv))
