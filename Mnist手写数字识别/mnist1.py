import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot=True)

x = tf.placeholder(tf.float32,[None,784])
y_ = tf.placeholder("float",[None,10])
w = tf.Variable(tf.truncated_normal(shape=[784,10],stddev=0.05))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,w)+b)#转化成10维的概率分布

cross_entropy = -tf.reduce_sum(y_*tf.log(y))#计算交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#梯度下降法优化交叉熵
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))#把布尔数组化成0与1并计算平均值
print (sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
