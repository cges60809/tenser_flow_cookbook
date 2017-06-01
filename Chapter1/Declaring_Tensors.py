#zero filled 
zero_var = tf.Variable(tf.zeros([row_dim, col_dim]))

#one filled
ones_var = tf.Variable(tf.ones([row_dim, col_dim]))

#tensor of similar shape
zero_similar = tf.Variable(tf.zeros_like(zero_var))
ones_similar = tf.Variable(tf.ones_like(ones_var))

#constant filled
fill_var = tf.Variable(tf.fill([row_dim, col_dim], -1))
const_var = tf.Variable(tf.constant([8, 6, 7, 5, 3, 0, 9]))
const_fill_var = tf.Variable(tf.constant(-1, shape=[row_dim, col_dim]))

# Linspace (Generates [0.0, 0.5, 1.0])
linear_var = tf.Variable(tf.linspace(start=0.0, stop=1.0, num=3))

# Range (Generates [6, 9, 12])
sequence_var = tf.Variable(tf.range(start=6, limit=15, delta=3)) 

# random
rnorm_var = tf.random_normal([row_dim, col_dim], mean=0.0, stddev=1.0)
runif_var = tf.random_uniform([row_dim, col_dim], minval=0, maxval=4)
