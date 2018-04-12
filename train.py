#coding=utf-8
#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#
#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# I used Ubuntu with Python 3.5, TensorFlow 1.0*, other OS should also be good.
# With current settings, 10000 traing steps needed 50 minutes on my laptop.

# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.

# Note: 
# it is suggested to restart your kenel to train the model multiple times 
#(in order to clear all the variables in the memory)
# Otherwise errors may occur: conv1/weights/biases already exist......


#%%

import os
import numpy as np
import tensorflow as tf
import input_data
import model
#%%

N_CLASSES = 6
IMG_W = 256  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 256
BATCH_SIZE = 8
CAPACITY = 10000
MAX_STEP = 800 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001

def run_training():
    
    # you need to change the directories to yours.
	s_train_dir = '/home/hrz/projects/tensorflow/emotion/ck+/CK+YuanTu'
	T_train_dir = '/home/hrz/projects/tensorflow/emotion/ck+/CK+X_mid'
	logs_train_dir = '/home/hrz/projects/tensorflow/emotion/ck+'
	s_train, s_train_label = input_data.get_files(s_train_dir)
	s_train_batch, s_train_label_batch = input_data.get_batch(s_train,
                                                          s_train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY)   
	T_train, T_train_label = input_data.get_files(T_train_dir)
    
	T_train_batch, T_train_label_batch = input_data.get_batch(T_train,
                                                          T_train_label,
                                                          IMG_W,
                                                          IMG_H,
                                                          BATCH_SIZE, 
                                                          CAPACITY) 

	train_logits = model.inference(s_train_batch,T_train_batch, BATCH_SIZE, N_CLASSES)
	train_loss = model.losses(train_logits, s_train_label_batch)        
	train_op = model.trainning(train_loss, learning_rate)
	train__acc = model.evaluation(train_logits, s_train_label_batch)
       
	summary_op = tf.summary.merge_all()  #汇总操作
	sess = tf.Session()   #定义sess
	train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph) #
	saver = tf.train.Saver()    #保存操作
    
	sess.run(tf.global_variables_initializer())#初始化所有变量
	coord = tf.train.Coordinator() #设置多线程协调器
	threads = tf.train.start_queue_runners(sess=sess, coord=coord) #开始Queue Runners(队列运行器)
    
    #开始训练过程
	try:
		for step in np.arange(MAX_STEP):
			if coord.should_stop():
					break
			_, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc]) 
               
			if step % 50 == 0:
				print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
				#运行汇总操作，写入汇总
				summary_str = sess.run(summary_op)
				train_writer.add_summary(summary_str, step)
            
			if step % 800 == 0 or (step + 1) == MAX_STEP:
				#保存当前模型和权重到 logs_train_dir,global_step为当前迭代次数
				checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
				saver.save(sess, checkpoint_path, global_step=step)
                
	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		coord.request_stop()
        
	coord.join(threads)
	sess.close()
run_training()

#%% Evaluate one image
# when training, comment the following codes.




from PIL import Image
import matplotlib.pyplot as plt

def get_one_image(train):
   '''Randomly pick one image from training data
   Return: ndarray
   '''
   n = len(train)
   ind = np.random.randint(0, n)
   img_dir = train[ind]

   image = Image.open(img_dir)
   plt.imshow(image)
   image = image.resize([208, 208])
   image = np.array(image)
   return image

def evaluate_one_image():
   '''Test one image against the saved models and parameters
   '''

   # you need to change the directories to yours.
   train_dir = '/home/kevin/tensorflow/cats_vs_dogs/data/train/'
   train, train_label = input_data.get_files(train_dir)
   image_array = get_one_image(train)

   with tf.Graph().as_default():
       BATCH_SIZE = 1
       N_CLASSES = 2

       image = tf.cast(image_array, tf.float32)
       image = tf.image.per_image_standardization(image)
       image = tf.reshape(image, [1, 208, 208, 3])
       logit = model.inference(image, BATCH_SIZE, N_CLASSES)

       logit = tf.nn.softmax(logit)

       x = tf.placeholder(tf.float32, shape=[208, 208, 3])

       # you need to change the directories to yours.
       logs_train_dir = '/home/kevin/tensorflow/cats_vs_dogs/logs/train/'

       saver = tf.train.Saver()

       with tf.Session() as sess:

           print("Reading checkpoints...")
           ckpt = tf.train.get_checkpoint_state(logs_train_dir)
           if ckpt and ckpt.model_checkpoint_path:
               global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
               saver.restore(sess, ckpt.model_checkpoint_path)
               print('Loading success, global_step is %s' % global_step)
           else:
               print('No checkpoint file found')

           prediction = sess.run(logit, feed_dict={x: image_array})
           max_index = np.argmax(prediction)
           if max_index==0:
               print('This is a cat with possibility %.6f' %prediction[:, 0])
           else:
               print('This is a dog with possibility %.6f' %prediction[:, 1])







