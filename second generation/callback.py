import tensorflow as tf
import os
from PIL import Image
import numpy as np
import pickle

class custom_callback(tf.keras.callbacks.Callback):
    
    def __init__(
        self,
        save_dir,
        checkpoint = 1, 
        save_dataset_path ='/content/save_dataset',
        save_model = True, 
        save_images = True,
    ):
        
        self.checkpoint = checkpoint #how many epochs will pass between saves
        self.save_dir = save_dir + "/saves"
        
        if not os.path.isdir(self.save_dir):
            os.mkdir(self.save_dir)

        self.will_save_model = save_model
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
        folders = os.listdir(self.save_dir)
        self.epoch = len(folders)
        
        self.log_steps = 0
        
        self.log_dir = save_dir + "/logs"
        log_dir = self.log_dir
        self.log_steps_save = self.log_dir + "/log_steps.pkl"
        if os.path.isfile(self.log_steps_save):
            self.log_steps = pickle.load(open(self.log_steps_save, "rb+"))
        
        
    
        
        self.summary_writers = {
            'd_loss': tf.summary.create_file_writer(log_dir +"/d_loss"),
            'g_loss': tf.summary.create_file_writer(log_dir +"/g_loss"),
            'd_loss_adv': tf.summary.create_file_writer(log_dir +"/d_loss_adv"),
            'd_loss_cls': tf.summary.create_file_writer(log_dir +"/d_loss_cls"),
            'd_loss_gp': tf.summary.create_file_writer(log_dir +"/d_loss_gp"),
            'g_loss_adv': tf.summary.create_file_writer(log_dir +"/g_loss_adv"),
            'g_loss_cls': tf.summary.create_file_writer(log_dir +"/g_loss_cls"),
            'image': tf.summary.create_file_writer(log_dir +"/image")
        }

    # generates sample images, has a significant runtime for 33  images
    def images(self, save_path):
        
        with self.summary_writers['image'].as_default():
          imgs = []
          for data in self.dataset:

              name , img , feature , noises = data

              name = name.numpy().decode("utf-8").split('.')[0]

              feature = feature.numpy()[0]
              feature = tf.constant([feature,feature,feature,feature,feature])
              noise = tf.concat([noises, feature], axis = -1)
              fake_images = self.model.generator(noise)  
              img = img[0]
              for f in fake_images:
                  img = tf.concat([img,f], axis = 1)
              
      
              img = ((img.numpy() * 255).astype("uint8"))
                  
              img = Image.fromarray(img)
              imgs.append(img)
              fl = open("{}/{}.png".format(save_path ,name) , "wb+")
              img.save(fl, format = "png")
              fl.close() 
              
          img_w, img_h = imgs[0].size
          offset = 15 # num of pixels between images
          background = Image.new('RGB',(img_w*3 + offset*2, img_h*11+ offset * 10), (255, 255, 255))
          images = iter(imgs)
          for y in range(11): 
              for x in range(3):
                  background.paste(next(images),( x*img_w + 15*x,y*img_h + 15*y))
                  
          fl = open("{}/{}.png".format(save_path ,"all") , "wb+")
          background.save(fl, format = "png")
          fl.close() 
          
          bg_w , bg_h = background.size
          as_tensor = tf.constant([np.array(background)/255])
          tf.summary.image('examples',as_tensor,step = self.epoch)
        
    def on_batch_end(self,batch ,logs = None): 
              
        with self.summary_writers['d_loss'].as_default():
            tf.summary.scalar('discriminator',logs['d_loss'][0][0],step = self.log_steps)
        with self.summary_writers['d_loss_adv'].as_default():
            tf.summary.scalar('discriminator',logs['d_loss_adv'][0][0],step = self.log_steps)
            tf.summary.scalar('discriminator vs generator',logs['d_loss_adv'][0][0],step = self.log_steps)
        with self.summary_writers['d_loss_cls'].as_default():
            tf.summary.scalar('discriminator',logs['d_loss_cls'][0][0],step = self.log_steps)
        with self.summary_writers['d_loss_gp'].as_default():
            tf.summary.scalar('discriminator',logs['d_loss_gp'][0][0],step = self.log_steps)
        with self.summary_writers['g_loss'].as_default():
            tf.summary.scalar('generator',logs['g_loss'][0][0],step = self.log_steps)
        with self.summary_writers['g_loss_adv'].as_default():
            tf.summary.scalar('generator',logs['g_loss_adv'][0][0],step = self.log_steps)
            tf.summary.scalar('discriminator vs generator',logs['g_loss_adv'][0][0],step = self.log_steps)
        with self.summary_writers['g_loss_cls'].as_default():
            tf.summary.scalar('generator',logs['g_loss_cls'][0][0],step = self.log_steps)
        self.log_steps +=1
        

        save_file = open(self.log_steps_save, "wb+")
        pickle.dump(self.log_steps, save_file)
                

    def on_epoch_end(self , epoch , logs = None):
       

        save_path = self.save_dir + "/epoch_{}".format(self.epoch)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        if self.will_save_model:
            self.model.save(save_path)


        if self.will_save_images:
            self.images(save_path)