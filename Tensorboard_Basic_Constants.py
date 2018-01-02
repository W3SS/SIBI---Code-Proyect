# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:32:42 2018

@author: Javier Carracedo

Grafo para representar en Tensorboad con operaciones de suma, resta, multiplicaciones y divisiones 
sobe constantes en Tensorflow.

"""

import tensorflow as tf

a = tf.constant(9,dtype=tf.float32,name='Constante1')
b = tf.constant(8,dtype=tf.float32,name='Constante2')
c = tf.constant(6,dtype=tf.float32,name='Constante3')
d = tf.constant(5,dtype=tf.float32,name='Constante4')


x = tf.add(a,b,name='Suma1')
y = tf.add(c,d,name='Suma2')


result0 = tf.add(x,y,name='SumaXY')         #Operacion de sumar
result1 = tf.subtract(x,y,name='RestaXY')   #Operacion de restar
result2 = tf.multiply(x,y,name='MultXY')    #Operacion de multiplicar
result3 = tf.divide(x,y,name='DivXY')       #Operacion de dividir.


resulFinal = tf.add(result0,result2, name='Final')  #Suma


with tf.Session() as sess:
 
    writter0 = tf.summary.FileWriter('./graphs',sess.graph) #Variable que recibe la operacion de creacion del grafo.
    
    
    print ("La suma de x=", sess.run(x) , ", e Y=",sess.run(y)," es ="," ",sess.run(result0))
    print ("La resta de x=", sess.run(x) , ", e Y=",sess.run(y)," es ="," ",sess.run(result1))
    print ("La multiplicacion de x=", sess.run(x) , ", e Y=",sess.run(y)," es ="," ",sess.run(result2))
    print ("La division de x=", sess.run(x) , ", entre Y=",sess.run(y)," es ="," ",sess.run(result3))
    print ("La suma de x + Y=", sess.run(result0) , ", más X por Y =",sess.run(result2)," es ="," ",sess.run(resulFinal))

    
writter0.close()

'''
    1- Para visualizar el grafo, simplemente accede al directorio en el que se encuentre este archivo
    a través de la CMD (terminal de Windows). Ejecuta el archivo a traves de la orden:
        
        >python Tensorboard_Basic_Constants.py
        
    2- Una vez que lo hayas ejecutado, sobre la misma terminal, ejecutas las siguiente instruccion para
    que se ejecute el programa que genera el grafo:
        
        >tensorboard --logdir="./graphs"
        
        
    3- Y finalmente, abres un navegador de internet (preferiblemente Google Chrome, ya que en otros puede
    dar fallos), y escribes en el navegador la siguiente URL:
        
         http://localhost:6006/
        
    4- Entre las numerosas pestañas que hay accedes a la que se llama "GRAPHS". ¡Y listo!
'''






