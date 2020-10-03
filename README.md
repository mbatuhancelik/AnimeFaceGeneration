Purpose: Tryin out different Gan architectures and messing with them during the process

first generation: 

Getting familiar with GAN's. 
Used https://www.kaggle.com/splcher/animefacedataset/notebooks as dataset, but sadly 64x64 waifus do not look pretty. 
Therefore, I abandonned this model and did not tried to further develop it. 

second generation: 

Helpful links: 
main architecture : https://nips2017creativity.github.io/doc/High_Quality_Anime.pdf 
classification : https://github.com/rezoo/illustration2vec 
pixel shuffle : https://github.com/apache/incubator-mxnet/issues/13548 : 
gradient penalty : https://medium.com/@jonathan_hui/gan-wasserstein-gan-wgan-gp-6a1a2aa1b490
gradient penalty implementation : https://keras.io/examples/generative/wgan_gp/#:~:text=Wasserstein%20GAN%20(WGAN)%20with%20Gradient%20Penalty%20(GP)&text=WGAN%20requires%20that%20the%20discriminator,space%20of%201-Lipschitz%20functions.

This generation is not trained yet, therefore, I cant share results. 

This repo could be useful only if I was able to implement the paper completely. Sadly, that's not the case and I could not understand the L_cls of lost function. Moreover, I was too lazy to check all of their references, so there are some hyperparameters I dont know. 










