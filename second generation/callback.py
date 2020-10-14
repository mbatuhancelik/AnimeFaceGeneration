import tensorflow as tf
import os
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class custom_callback(tf.keras.callbacks.Callback):
    
    def __init__(
        self,
        save_dir,
        checkpoint = 1, 
        save_dataset_path ='F:\\messy code\\afg ordered\\second generation\\save_dataset',
        save_model = True, 
        plot_losses = False, 
        save_images = True
    ):
        
        self.checkpoint = checkpoint #how many epochs will pass between saves
        self.save_dir = save_dir

        self.will_save_model = save_model
        self.will_plot_losses = plot_losses
        self.will_save_images = save_images

        name_spec = tf.TensorSpec((None) , dtype = tf.string)
        img_spec = tf.TensorSpec((None,128,128,3) , dtype=tf.float32)
        feature_spec = tf.TensorSpec((1 , 39) , dtype=tf.float32)
        noise_spec = tf.TensorSpec((5,128), dtype = tf.float32)
        self.dataset = tf.data.experimental.load(
                                            save_dataset_path,
                                            element_spec = (name_spec, img_spec,feature_spec, noise_spec),
                                            compression='GZIP', 
                                            reader_func=None )

        folders = os.listdir(save_dir)
        self.epoch = len(folders)
        self.x_axis = []
        self.losses={}
        
        if self.epoch != 0: 
            last_folder = folders.pop()
            
            last_losses_save = '{}\\{}\\losses'.format(save_dir, last_folder)
            losses = os.listdir(last_losses_save)
            
            for loss in losses: 
                self.losses[loss.split('.')[0]] = np.load("{}\\{}".format(last_losses_save, loss))
            self.x_axis = [x for x in range (len(self.losses['d_loss']))]

    # generates sample images and displays them, has a significant runtime for 33 images
    def images(self, save_path):
        plt.figure(figsize=(11,17),dpi = 300)
        

        counter = 1 
        for data in self.dataset:

            name , img , feature , noises = data

            imgs = img

            plt.subplot( 17,2 ,counter)
            plt.axis("off")
            counter += 1

            #Note: generating images using a (5,167) noise could be faster, remember that if you gonna try to get 
            #one epoch train in less than an hour
            for noise in noises: 

                noise = tf.concat([tf.reshape(noise , (1,128)), feature], axis = -1)
                img = self.model.generator(noise)          
                imgs = tf.concat([imgs , img] , axis = 2)
    
            imgs = ((imgs.numpy() * 255).astype("uint8"))[0]
            plt.imshow(imgs, aspect='auto')
            imgs = Image.fromarray(imgs)
            fl = open("{}\\{}.png".format(save_path ,name.numpy().decode("utf-8").split('.')[0]) , "wb+")
            imgs.save(fl, format = "png")
            fl.close()            
        plt.savefig('{}\\waifus.png'.format(save_path),dpi = 300)
        #plt.show()
        

    def plot_shit(self , keywords, subplot):

        colors = iter(['#1f77b4','#ff7f0e','#2ca02c','#d62728'])
        lines = []
        for keyword in keywords: 
            a, = subplot.plot(self.x_axis , self.losses[keyword],next(colors), label = keyword)
            lines.append(a)
        subplot.legend(handles = lines, loc = 'best')
        
    def plot_losses(self): 
        
        self.plot_shit(['d_loss_adv', 'g_loss_adv'],self.ax1)
        self.plot_shit(['d_loss', 'g_loss'],self.ax2)
        self.plot_shit(['g_loss','g_loss_adv', 'g_loss_cls'],self.ax3)
        self.plot_shit(['d_loss','d_loss_adv','d_loss_cls','d_loss_gp'],self.ax4)
        
        
        self.losses_figure.canvas.draw()

    def on_train_begin(self, logs = None): 
        
        self.losses_figure, (self.ax1,self.ax2,self.ax3,self.ax4)= plt.subplots(4, 1, dpi = 100, figsize= (10,20))
        plt.ion()
        if self.will_plot_losses:
            plt.show()

    def on_batch_end(self,batch ,logs = None): 
        self.x_axis.append(len(self.x_axis))

        test = not self.losses
        if test: 
            for log in logs: 
                self.losses[log] = logs[log][0]
        else: 
            for log in logs:
                self.losses[log] = np.append(self.losses[log],logs[log][0][0])
                
            if self.will_plot_losses:
                self.plot_losses()
                
    def on_epoch_end(self , epoch , logs = None):
        self.epoch += 1
        
        if self.epoch % self.checkpoint != 0:
            return
       

        save_path = self.save_dir + "\\epoch_{}".format(self.epoch)
        os.mkdir(save_path)
        
        if self.will_save_model:
            self.model.save(save_path)


        if self.will_save_images:
            self.images(save_path)
        
        
        save_losses_path = "{}\\losses".format(save_path)
        os.mkdir(save_losses_path)
        
        for loss in self.losses:
            np.save('{}\\{}'.format(save_losses_path,loss), self.losses[loss])
   