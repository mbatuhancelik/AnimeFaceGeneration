Purpose: Tryin out different Gan architectures and messing with them during the process

first generation: 

Getting familiar with GAN's. 
Used https://www.kaggle.com/splcher/animefacedataset/notebooks as dataset, but sadly 64x64 waifus do not look pretty. 
Therefore, I abandonned this model. 

second generation: 

  Helpful links: 

    classification: https://github.com/rezoo/illustration2vec 
    pixel shuffle: https://github.com/apache/incubator-mxnet/issues/13548
    gradient penalty: https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
    gradient penalty implementation:  https://keras.io/examples/generative/wgan_gp/#:~:text=Wasserstein%20GAN%20(WGAN)%20with%20Gradient%20Penalty%20(GP)&text=WGAN%20requires%20that%20the%20discriminator,space%20of%201-Lipschitz%20functions.

Reimplementation of https://nips2017creativity.github.io/doc/High_Quality_Anime.pdf. Sadly it is not trained enough to share the results, yet after 50 epochs, it generates some results. After 50 epochs I got bored and tried different loss functions for fun. 
