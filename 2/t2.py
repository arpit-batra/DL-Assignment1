import tensorflow as tf #importing the tensorflow library
import numpy as np
import math
from numpy import *
import copy

# loading the input files
masses=np.load('q2_input/masses.npy')
positions=np.load('q2_input/positions.npy')
velocities=np.load('q2_input/velocities.npy')

num_of_particles=100

m = tf.Variable(masses, dtype=tf.float32, name='m')
p = tf.Variable(positions, dtype=tf.float32, name='p')
v = tf.Variable(velocities, dtype=tf.float32, name='v')
G = tf.constant((-6.67*math.pow(10,5)), name='G')

threshold = 0.1
time = tf.constant(0.0001)

x2 = tf.reshape(p,(num_of_particles,1,2), name='x2')
x3 = tf.subtract(x2,p, name='sub1') # R, full position matrix  x3-x2 where each i,jth cell is showing distance of j from ith particle

m1 = tf.reshape(m,(num_of_particles,1,1), name='m1') # reshaped m  (1,num_of_particless,1)
x4 = tf.multiply(x3,m1, name='m1_mul_r12')         #m*R   

x5=tf.linalg.norm(x3, axis=-1, name='norm_x5') # R, full position matrix with mod of each element
x5=tf.math.pow(x5,3, name='norm_cube_x5')      # R, full position matrix with cube of mod of each element
x5=tf.reshape(x5,(num_of_particles,num_of_particles,1), name='norm_cube_reshape_x5')

x6=tf.divide(x4,x5, name='divide_r12_div_r12_ka_cube')    #m*R/|R|^3 
x6=tf.where(tf.is_nan(x6), tf.zeros_like(x6), x6, name='check_nan')
x6= tf.multiply(x6,G, name='x6_mul_G')    #R, full position matrix  x3-x2 where each i,jth cell is showing acceleration on i due to i
x7 = tf.reduce_sum(x6, 0, name='col_wise_sum_a') # row wise acceleration matrix ka vector sum pehle 1 tha

# # #threshold:
x8=tf.linalg.norm(x3, axis=-1)
diag = tf.Variable(np.ones(num_of_particles), dtype=tf.float32, name='diag')
x8=tf.linalg.set_diag(x8,diag, name='mod_vector_matrix')
minimum_distance= tf.math.reduce_min(x8)

# finally calculating new position and velocity
x9 = tf.add(p,tf.multiply(time,v))
x9 = tf.add(x9,tf.multiply(x7,(time*time/2)), name='x9') # new position matrix 100*2
x10 = tf.add(v,tf.multiply(time,x7), name='x10') # new velocity matrix 100*2

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    a = sess.run(minimum_distance)
    epoch=1
    while(a>threshold):
        # # updating new position and velocity to older arrays
        a, p_new, v_new = sess.run([minimum_distance,x9,x10])
        sess.run(tf.assign(p,p_new))
        sess.run(tf.assign(v,v_new))
        print epoch,' ----- ',a
        epoch+=1
    
    # np.save('Final_coordinates_of_all_particles', p_new)
    # writer = tf.summary.FileWriter("logs",sess.graph)

