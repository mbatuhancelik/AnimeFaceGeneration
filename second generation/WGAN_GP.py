import tensorflow as tf
# tutorial : https://keras.io/examples/generative/wgan_gp/#:~:text=Wasserstein%20GAN%20(WGAN)%20with%20Gradient%20Penalty%20(GP)&text=WGAN%20requires%20that%20the%20discriminator,space%20of%201-Lipschitz%20functions.&text=Instead%20of%20clipping%20the%20weights,discriminator%20gradients%20close%20to%201.
class WGAN(tf.keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        discriminator_lambda_adv = 3, 
        discriminator_lambda_cls = 5,
        discriminator_lambda_gp = 1,
        generator_lambda_adv = 1,
        generator_lambda_cls = 4,
        discriminator_steps=3,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.discriminator_steps = discriminator_steps

        self.discriminator_lambda_adv = tf.constant(discriminator_lambda_adv, dtype = tf.float32 , shape = (1,1))
        self.discriminator_lambda_cls = tf.constant(discriminator_lambda_cls, dtype = tf.float32 , shape = (1,1))
        self.discriminator_lambda_gp = discriminator_lambda_gp

        self.generator_lambda_adv = tf.constant(generator_lambda_adv, dtype = tf.float32 , shape = (1,1))
        self.generator_lambda_cls = tf.constant(generator_lambda_cls, dtype = tf.float32 , shape = (1,1))
        

    def compile(self, d_optimizer, g_optimizer, d_loss, g_loss):
        super(WGAN, self).compile()
        self.discriminator_optimizer = d_optimizer
        self.generator_optimizer = g_optimizer
        self.discriminator_loss = d_loss
        self.generator_loss = g_loss

    def gradient_penalty(self,batch_size, real_images, fake_images):
        
        # get the interplated image

        #batch_size = real_images.shape[0]
        alpha = tf.random.normal((batch_size, 1, 1, 1), 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training = True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calcuate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self , batch):
        assert(len(batch) == 2)
        real_images = batch[0]
        real_classes = batch[1]
        
        class_size = real_classes.shape[-1]

        
        noise = tf.random.uniform([1, 128])
        noise = tf.concat([noise , real_classes], axis = -1)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = self.generator(noise , training = True)
            # discrimination and applying discrimination loss
            for i in range(self.discriminator_steps):
            
                fake_results = self.discriminator(fake_images , training = True)

                real_results = self.discriminator(real_images , training = True)

                d_cost, discriminator_L_adv , discriminator_L_cls = self.discriminator_loss(
                                                                                            real = real_results ,
                                                                                            fake = fake_results , 
                                                                                            real_tags = real_classes, 
                                                                                            lambda_adv = self.discriminator_lambda_adv,
                                                                                            lambda_cls = self.discriminator_lambda_cls 
                                                                                            )
                gp = self.gradient_penalty( 1, real_images , fake_images)

                d_loss  = d_cost + self.discriminator_lambda_gp * gp
            d_gradient = disc_tape.gradient(d_loss , self.discriminator.trainable_variables)
            self.discriminator_optimizer.apply_gradients(
                zip(d_gradient , self.discriminator.trainable_variables)
            )
            # applying generator loss

            g_loss , generator_L_adv , generator_L_cls = self.generator_loss(fake_results,real_classes , self.generator_lambda_adv, self.generator_lambda_cls)

            gen_gradient = gen_tape.gradient(g_loss , self.generator.trainable_variables)
            self.generator_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
            )
        return {"d_loss": d_loss,"d_loss_adv" :discriminator_L_adv , "d_loss_cls" :discriminator_L_cls,
                 "g_loss": g_loss, "g_loss_cls" :generator_L_adv , "g_loss_adv" :generator_L_cls }