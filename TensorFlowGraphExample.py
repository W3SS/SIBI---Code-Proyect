# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 15:30:01 2018

@author: ASUS
"""

import tensorflow as tf

##Create a graph

g = tf.Graph()

with g.as_default():
    
    x = tf.placeholder(dtype=tf.float32,
                       shape=None,name='z')
    w = tf.Variable(2.0, name='weight')
    
    b = tf.Variable(0.7,name='bias')
    
    z = w*x + b
    
    init = tf.global_variables_initializer()
    
##Create a session and pass in graph g
    
with tf.Session(graph=g)as sess:
    
    
    writter0 = tf.summary.FileWriter('./graphs',sess.graph)
    
    #initialize w and b:
    
    sess.run(init)
    
    ## evaluate z:
    
    for t in [1.0,0.6,-1.8]:
        print('x=%4.1f --> z=%4.1f'%(t, sess.run(z, feed_dict={x:t})))

writter0.close()

print("Run in your cmd: >python TensorFlowGraphExample.py")
print("Run: >tensorboard --logdir=./graphs")