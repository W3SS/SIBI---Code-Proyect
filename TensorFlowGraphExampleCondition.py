# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 11:32:37 2018

@author: Javier Carracedo
"""

import tensorflow as tf

x, y = 1.0,2.0

g = tf.Graph()

with g.as_default():
    
    tf_x = tf.placeholder(dtype=tf.float32,
                           shape=None,
                           name='tf_x')
    
    tf_y = tf.placeholder(dtype=tf.float32,
                          shape=None,
                          name='tf_y')
    
    res = tf.cond(tf_x < tf_y, 
                  lambda: tf.add(tf_x,tf_y,
                                        name='result_add'),
                  lambda: tf.subtract(tf_x,tf_y,
                                        name='result_sub'))
    print('Object',res)
    
with tf.Session(graph=g) as sess:
    #init = tf.global_variables_initializer()

    #sess.run(init)
    
    writter0 = tf.summary.FileWriter(logdir= './graphs',graph= g)
    
    print('x < y: %s -> Result:' % (x<y),
          res.eval(feed_dict={'tf_x:0': x,
                             'tf_y:0': y}))
    x,y = 2.0,1.0   
    
    print('x < y: %s -> Result:' % (x<y),
          res.eval(feed_dict={'tf_x:0': x,
                             'tf_y:0': y}))
              

#writter0.close()             
                  
    
    
                    
