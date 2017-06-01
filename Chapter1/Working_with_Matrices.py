#Identity Matrix
identity_matrix = tf.diag([1.0,1.0,1.0])
#2x3 constant matrix
B = tf.fill([2,3], 5.0)
#3x2 random uniform matrix
C = tf.random_uniform([3,2])
#Create matrix from np array
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
"""
Matrix Operations
"""
#Matrix addition/subtraction
sess.run(B+B)
sess.run(B-B)
#Matrix Transpose
sess.run(tf.transpose(C))
#Matrix Determinant
sess.run(tf.matrix_determinant(D))
#Matrix Inverse
sess.run(tf.cholesky(identity_matrix))
#Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = sess.run(tf.self_adjoint_eig(D))
