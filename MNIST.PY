# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 10:58:11 2018

@author: Javier Carracedo
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("MNIST_data/", one_hot=True) #La imagenes tienen dimension de 28x28

    
x=tf.placeholder(tf.float32,[None,784],name='x') #imagen del numero descompuesta a un vector
P=tf.Variable(tf.zeros([784,10]),name='p') #Matriz de pesos, 784 para recibir la imagen, 10 por las posible salidas
b=tf.Variable(tf.zeros([10]),name='b') #Vector con bias
y=tf.matmul(x,P)+b #La operacion que se hara en los nodos que reciben entradas
yR=tf.placeholder(tf.float32,[None,10],name='yR') # Matriz con las etiquetas REALES del set de datos

'''
Definir la función de costo entropia cruzada (Cross Entropy) para poder medir el error. La salida será con Softmax¶
'''

softmax=tf.nn.softmax_cross_entropy_with_logits(labels=yR,logits=y)
costo=tf.reduce_mean(softmax)
optimizador=tf.train.GradientDescentOptimizer(0.5).minimize(costo)

'''Correr la gráfica computacional'''

prediccion = tf.equal(tf.argmax(y, 1), tf.argmax(yR, 1)) #Nos da arreglo de booleanos para decirnos 
                                                         #cuales estan bien y cuales no
accuracy = tf.reduce_mean(tf.cast(prediccion, tf.float32))#Nos da el porcentaje sobre el arreglo de prediccion
Produccion = tf.argmax(y,1)
init=tf.global_variables_initializer()

'''Entrenar algoritmo'''

#Funcion que usaremos para ver que tan bien va a aprendiendo nuestro modelo 
def avance(epoca_i, sess, last_features, last_labels):
    costoActual = sess.run(costo,feed_dict={x: last_features, yR: last_labels})
    Certeza = sess.run(accuracy,feed_dict={x:mnist.validation.images,yR: mnist.validation.labels})
    print('Iteraccion: {:<4} - Costo: {:<8.3} Precision: {:<5.3}'.format(epoca_i,costoActual,Certeza))
    
    
with tf.Session() as sess:
    
    writter0 = tf.summary.FileWriter('./graphs',sess.graph)
    
    sess.run(init)
    for epoca_i in range(1000):
        lotex, lotey = mnist.train.next_batch(100)
        sess.run(optimizador, feed_dict={x: lotex, yR: lotey})
        if (epoca_i%50==0):
            avance(epoca_i, sess, lotex, lotey)
    print('RESULTADO FINAL: ',sess.run(accuracy, feed_dict={x: mnist.test.images,yR: mnist.test.labels}))
    #print ('Resultado de una imagen',sess.run(Produccion,feed_dict={x: mnist.test.images[5].reshape(1,784)}))
    

    writter0.close()