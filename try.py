import numpy as np
import tensorflow as tf

# Model parameters

dim1=3
dim2=1
dim3=dim2
dim4= dim1
dim5=1

W = tf.Variable(tf.zeros([dim1,dim2]), dtype=tf.float32)   
b = tf.Variable(tf.zeros([dim3]), dtype=tf.float32)         


# Model input and output
x = tf.placeholder(tf.float32,[None,dim4])  
y = tf.placeholder(tf.float32,[None,dim5])  

y_model = tf.matmul(x, W) + b

# loss

M=(y_model-y)
loss = tf.reduce_sum(tf.square( M )) # sum of the squares


optimizer = tf.train.GradientDescentOptimizer(0.0001)
train = optimizer.minimize(tf.reduce_sum(tf.square( M )))


# x_train = np.load("x_train.npy")
# y_train = np.load("y_train.npy")

x_train = np.zeros((1000,3))
for i in range(1000):
    for j in range(3):
        x_train[i][j]=np.random.rand()
ans = [[0.1],[0.2],[0.3]]
y_train = np.matmul(x_train,ans)


# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(2500):
    if i % 25 ==0:
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
        # print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))


print ("predicted time for this user will be")
res = np.matmul([10,20,30],curr_W)+curr_b
print(res)
