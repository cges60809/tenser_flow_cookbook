#Arithmetic Operations
sess.run(tf.div(3,4))
sess.run(tf.truediv(3,4))
sess.run(tf.floordiv(3.0,4.0))
sess.run(tf.mod(22.0,5.0))
sess.run(tf.cross([1.,0.,0.],[0.,1.,0.]))

#Trig functions
sess.run(tf.sin(3.1416))
sess.run(tf.cos(3.1416))
sess.run(tf.div(tf.sin(3.1416/4.), tf.cos(3.1416/4.)))

#Custom operations
test_nums = range(15)

def custom_polynomial(x_val):
    # Return 3x^2 - x + 10
    return(tf.subtract(3 * tf.square(x_val), x_val) + 10)

expected_output = [3*x*x-x+10 for x in test_nums]

for num in test_nums:
    sess.run(custom_polynomial(num))
    
