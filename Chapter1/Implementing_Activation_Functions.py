"""
Activation Functions
"""
#ReLU activation
sess.run(tf.nn.relu([-3., 3., 10.]))

#ReLU-6 activation
sess.run(tf.nn.relu6([-3., 3., 10.]))

#Sigmoid activation
sess.run(tf.nn.sigmoid([-1., 0., 1.]))

#Hyper Tangent activation
sess.run(tf.nn.tanh([-1., 0., 1.]))

#Softsign activation
sess.run(tf.nn.softsign([-1., 0., 1.]))

#Softplus activation
sess.run(tf.nn.softplus([-1., 0., 1.]))

#Exponential linear activation
sess.run(tf.nn.elu([-1., 0., 1.]))

