# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:32:42 2018

@author: Javier Carracedo

Grafo para representar en Tensorboad con operaciones de suma, resta, multiplicaciones y divisiones 
sobre placeholders en Tensorflow.

"""

import tensorflow as tf

Valor_a = tf.placeholder(dtype=tf.float32,shape=(1),name='a')
Valor_b = tf.placeholder(dtype=tf.float32,shape=(1),name='b')
Valor_c = tf.placeholder(dtype=tf.float32,shape=(1),name='c')
Valor_d = tf.placeholder(dtype=tf.float32,shape=(1),name='d')




with tf.Session() as sess:
 
   

    a = sess.run(Valor_a,feed_dict={Valor_a:[9]})
    b = sess.run(Valor_b,feed_dict={Valor_b:[8]})
    c = sess.run(Valor_c,feed_dict={Valor_c:[6]})
    d = sess.run(Valor_d,feed_dict={Valor_d:[5]})
    
    
    x = tf.add(a,b,name='Suma1')
    y = tf.add(c,d,name='Suma2')
    
    
    result0 = tf.add(x,y,name='SumaXY')         #Operacion de sumar
    result1 = tf.subtract(x,y,name='RestaXY')   #Operacion de restar
    result2 = tf.multiply(x,y,name='MultXY')    #Operacion de multiplicar
    result3 = tf.divide(x,y,name='DivXY')       #Operacion de dividir.


    resulFinal = tf.add(result0,result2, name='Final')  #Suma
    
    writter0 = tf.summary.FileWriter('./graphs',sess.graph) #Variable que recibe la operacion de creacion del grafo.
    
    
    
    print ("La suma de x=", sess.run(x) , ", e Y=",sess.run(y)," es ="," ",sess.run(result0))
    print ("La resta de x=", sess.run(x) , ", e Y=",sess.run(y)," es ="," ",sess.run(result1))
    print ("La multiplicacion de x=", sess.run(x) , ", e Y=",sess.run(y)," es ="," ",sess.run(result2))
    print ("La division de x=", sess.run(x) , ", entre Y=",sess.run(y)," es ="," ",sess.run(result3))
    print ("La suma de x + Y=", sess.run(result0) , ", más X por Y =",sess.run(result2)," es ="," ",sess.run(resulFinal))

    
writter0.close()
