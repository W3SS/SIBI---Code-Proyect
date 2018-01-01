# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 23:24:34 2018

@author: Javier Carracedo  - Universidad de Leon
"""

# -*- coding: utf-8 -*-

'''
    
    TensorFlow: Variables y sesiones.
    
        - Variables.
        - GrÃ¡fica comutacional.
        - Sesiones.
    
'''


import tensorflow as tf


'''
1 - Constante: Constante que no va a cambiar su valor en todo el flujo del programa.
    
    * Primer Argumento: Definimos el valor que queremos que tome la constante. En este caso
                        un vector con tres elementos.
                        
    * Segundo Argumento: Decimos que tipo de dato es. En este caso hemos definido un float32.
                         Mas concretamente un vector de decimales.
                        
    * Tercer Argumento: El nombre que queremos darle a la variable. En este caso 'Constante1'.


    NOTA: Con esta sentencia solo estamos definiendo la "grÃ¡fica computacional". Es decir, al ejecutar
          la siguiente instrucciÃ³n, todavÃ­a no tendrÃ¡ los valores definidos. Por eso, si utilizaramos la 
          instrcucion print(constante) el resultado que nos devolverÃ­a serÃ­a este:

              Tensor("Constante1_1:0", shape=(3,), dtype=float32)
'''

constante =tf.constant([1.1,3,4],dtype=tf.float32, name= 'Constante1')
#print(constante)

'''
2 - Placeholder - Se define asÃ­ a un tipo de variables simbÃ³licas. Que al principio, en la ejecuciÃ³n del
                  programa estarÃ¡n vacÃ­as y posteriormente se irÃ¡n "llenando" conforme se vaya ejecutando
                  el cÃ³digo. Normalmente serÃ¡n usados para nuestros inputs de data.
                  
    * Primer Argumento (dtype) : Definimos tipo de valor que queremos que tome el placeholder. En este
                                 en esta caso float32.
                                 
                        
    * Segundo Argumento (shape) : Decidimos la forma que queremos que tome el placeholder. Es opcional. Si no
                                  estÃ¡ especificado, puede ser alimnetado un tensor de cualquier forma. 
                                  En este caso una matriz de 2 filas y 3 columnas.
                        
    * Tercer Argumento (name) : El nombre que queremos darle al placeholder. En este caso 'apartado1'.


    NOTA: Al igual que con la constante anterior, solamente hemos definido la "grÃ¡fica computacional", si
          intentÃ¡semos imprimir por pantalla con la variable 'apartado' con ayuda de la funciÃ³n print(apartado), el
          resultado que obtendrÃ­amos serÃ­a este: 
    
              Tensor("Apartado1_1:0", dtype=float32)

'''
apartado = tf.placeholder(tf.float32,shape=(2,3),name='Apartado1')
#print(apartado)                
                  
'''
3 - Variable: Es un tipo dato el cual podrÃ¡ cambiar su valor a lo largo de la ejecuciÃ³n del programa. Es muy 
              parecida a la variable constante, con la Ãºnica diferencia que esta Ãºltima no puede cambiar su
              valor a lo largo de la ejecuciÃ³n del programa. En el tipo de dato Variable si podremos cambiar 
              su valor durante la ejecuiÃ³n del programa. Aunque se le pueden asignar varios argumentos a la 
              hora de crear una varialbe, los mÃ¡s importantes son estos:
              
                  
     * Primer Argumento (valor) : Definimos el valor que queramos que tenga la variable. En este caso 3.
                     
            
     * Segundo Argumento (dtype) : Definimos el tipo de dato que queremos que tenga. En este caso float32
            
     * Tercer Argumento (name) : El nombre que queremos darle a la variable. En este caso 'variable1'.
              
     NOTA: Al igual que con el placeholder anterior, solamente hemos definido la "grÃ¡fica computacional", si
              intentÃ¡semos imprimir por pantalla con la variable 'variable' con ayuda de la funciÃ³n print(variable), el
              resultado que obtendrÃ­amos serÃ­a este: 
                  
                  <tf.Variable 'variable1:0' shape=() dtype=float32_ref>
              
'''                 
variable = tf.Variable(3, dtype=tf.float32, name='variable1')
#print(variable)      


'''
3 - Matriz de Zeros: Es una matriz de ceros que se usarÃ¡ para inicializar muchas veces para implementar 
                     al inicio una red neuronal. Al definir el modelo de la red.
                     
     * Primer Argumento (shape): Es la forma que queremos que tenga la matriz. En este caso una matriz
                                 de 3 filas y tres columnas.
     * Segundo Argumento (dtype): El tipo de dato que queremos que tenga la matriz.En este caso hemos
                                  decidido que tenga un float32.
     * Tercer Argumento (name): El nombre que queremos darle a la variable. En este caso 'matriz'.
     
     NOTA: Al igual que con el placeholder anterior, solamente hemos definido la "grÃ¡fica computacional", si
              intentÃ¡semos imprimir por pantalla con la variable 'variable' con ayuda de la funciÃ³n print(variable), el
              resultado que obtendrÃ­amos serÃ­a este: 
                  
                  Tensor("matriz_2:0", shape=(3, 4), dtype=float32)
'''       
matriz = tf.zeros([3,4],dtype=tf.float32,name='matriz')
#print(matriz)



'''

    GrÃ¡fica computacional: Se podrÃ­a decir que son como los planos de nuestro cÃ³digo. Esto es debido a que
                           la forma de trabajar de TensorFlow, es un poco diferente al del resto del lenguajes.
                           En este caso, aunque estamos trabajando con el lenguaje Python. Se podrÃ­a decir que
                           la forma con la que se operan y realizan las operaciones son un poco especiales.
                           
                           Por eso la forma de trabajar de TensorFlow es espcificar primero todo el problema, y al
                           final crear una sesiÃ³n (cÃ³mo se verÃ¡ a continuaciÃ³n). Para finalemente, ejecutar la
                           computaciÃ³n asociada a las instrucciones.

'''

'''
    Sesiones - Para poder ejecutar la grÃ¡fica computacinal anteriormente especificada, tendremos que
               utilizar las sesiones.
               
               Para ello lo primero que se ha de hacer es inicializar todas las variables. A travÃ©s de la
               instrucion: 
               
                   inicializar = tf.global_variables_initializer()
               
               A continuaciÃ³n se le asigna a una variable una sesion, este puede ser un ejemplo:
                   
                   sess = tf.Session()
               
               Y para que finalmente la grÃ¡cfica computacional corra, ejecutamos:
                   
                   sess.run(inicializar)
                   
               Una vez ejecutado la instrucion anterior, la grÃ¡fica computacinal ya corriÃ³. Ahora, todas las
               variables tienen un valor asignado.
               
            
'''
inicializar = tf.global_variables_initializer()

sess = tf.Session()

sess.run(inicializar)


#print(sess.run(constante))

'''
    Ejemplo de multiplicaciÃ³n de matrices (Recordando que para la multiplicaciÃ³n de matrices
    de la forma (M,N)x(N,M), el resultado serÃ¡ de dimensiÃ³n (M,M)
    
'''




a = tf.placeholder(tf.float32,shape=(2,2))
b = tf.placeholder(tf.float32,shape=(2,3))
mult = tf.matmul(a,b) #Operacion de la multiplicacion de las matrices a y b.
init = tf.global_variables_initializer() #Instrucción que inicializa nuestras variables
sess = tf.Session() #Para poder empezar a ejecutar nuestra gráfica computacinal.
sess.run(init)


'''
Queremos que imprima por pantalla la variable 'mult' que es la ejecucion de la multiplicacion de matrices.
Dado que las operaciones que queremos hacer sobre las matrices 'a' y 'b' son placesholders, no tinen asignados valores.
Con la instruccion feed dict se alimentan dichos placesholders.
'''

print("RESULTADO MULT A & B: ","\n",sess.run(mult,feed_dict={ a:[[1,2],[2,2]], b:[[12,21,4],[3,2,4]]}))

'''
    Producto - Punto a punto de dos vectores.
'''


c = tf.placeholder(tf.float32, shape=(3))
d = tf.placeholder(tf.float32, shape=(3))
punto = tf.tensordot(c,d, 1)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print("RESULTADO MULT C & D: ",sess.run(punto, feed_dict={c:[1,2,3],d:[3,2,1]}))




