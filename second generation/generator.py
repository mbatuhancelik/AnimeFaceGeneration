# shuffler block implementation : https://github.com/atriumlts/subpixel
# res block tutorial: https://towardsdatascience.com/hitchhikers-guide-to-residual-networks-resnet-in-keras-385ec01ec8ff
import tensorflow as tf
from tensorflow.keras.layers import Conv2D , BatchNormalization , ReLU , Add , Dense
from tensorflow.keras import backend as K
def residual_block(X):
    
    X_shortcut = X
    # TODO : fill paranthesis
    X = Conv2D(filters = 64, kernel_size = (3,3) , strides = (1,1),padding = 'same')(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Conv2D(filters = 64, kernel_size = (3,3) , strides = (1,1), padding = 'same')(X)
    X = BatchNormalization()(X)
    X = Add()([X , X_shortcut])

    return X

def ps(I , r):
        bsize, a, b, c = I.get_shape().as_list()
        bsize = K.shape(I)[0] # Handling Dimension(None) type for undefined batch dim
        X = K.reshape(I, [bsize,int( a), int(b), int(c/(r*r)),int(r), int(r)]) # bsize, a, b, c/(r*r), r, r
        X = K.permute_dimensions(X, (0, 1, 2, 5, 4, 3))  # bsize, a, b, r, r, c/(r*r)
        #Keras backend does not support tf.split, so in future versions this could be nicer
        X = [X[:,i,:,:,:,:] for i in range(a)] # a, [bsize, b, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, b, a*r, r, c/(r*r)
        X = [X[:,i,:,:,:] for i in range(b)] # b, [bsize, r, r, c/(r*r)
        X = K.concatenate(X, 2)  # bsize, a*r, b*r, c/(r*r)
        return X

def pixel_shuffler(X , r):
    Xc = tf.split(X, 3, 3)
    X = tf.concat( [ps(x, r) for x in Xc], 3)
    return X
def ps_block(X , r):
    X = Conv2D(filters = 128*3, kernel_size = (3,3) , strides = (1,1), padding = 'same')(X)
    X = pixel_shuffler(X , r)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    return X
def generator_model(input_shape = ( 128 + 39)): 

    X_input = tf.keras.Input((input_shape,))

    X = tf.keras.layers.Dense(16*16*64 , activation = 'relu' ,  kernel_initializer='glorot_uniform')(X_input)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = tf.keras.layers.Reshape(target_shape = (16,16,64))(X)
    X_short = X
    for i in range(16): 
        X = residual_block(X)
    X = Conv2D(filters = 64, kernel_size = (3,3) , strides = (1,1), padding = 'same')(X)
    X = BatchNormalization()(X)
    X = ReLU()(X)
    X = Add()([X, X_short])

    for i in range(3): 
        X = ps_block(X, 2)
    
    X = Conv2D(filters = 3, kernel_size = (9,9) , strides = (1,1), padding = 'same')(X)
    X = tf.keras.activations.sigmoid(X)
    print('final {}'.format(X))
    model = tf.keras.Model(inputs = X_input, outputs = X , name = 'generator')
    
    return model

def generator_loss(fake , real_tags , lambda_adv, lambda_cls):
    fake_genuity , fake_attributes = fake[0] , fake[1]
    negative_one = tf.constant(-1 , dtype = tf.float32 , shape = (1,1))

    L_adv = tf.math.log(fake_genuity)
    L_adv = tf.math.multiply(L_adv , negative_one)
    L_adv = tf.math.multiply(L_adv , lambda_adv)
    L_adv = tf.math.reduce_sum(L_adv , axis = 0, keepdims = True)
    

    #Sadly, I lack the mathematical background to understand how L_cls in the paper works. 
    #This is just the distance between vectors
    L_cls = tf.math.squared_difference(real_tags ,fake_attributes)
    L_cls = tf.math.reduce_sum(L_cls , axis = -1 , keepdims = True)
    L_cls = tf.sqrt(L_cls)
    L_cls = tf.math.multiply(L_cls , lambda_cls)
    L_cls = tf.math.reduce_sum(L_cls, axis = 0, keepdims = True)
    

    return tf.math.add(L_cls , L_adv) , L_adv,L_cls
