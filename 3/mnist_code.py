import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

sess = tf.Session()


def dense_layer(input, image_size, num_class):
	w = tf.Variable(tf.ones([image_size, num_class]))
	b = tf.Variable(np.random.randn(num_class),dtype=tf.float32)
	z = tf.matmul(input, w) + b
	print z
	a = tf.nn.softmax(z)
	y_pred_class = tf.argmax(a, axis=1)
	return z, y_pred_class

def dense_layer_2(input,image_size, num_class):
    w = tf.Variable(tf.random_normal([image_size, num_class],dtype=tf.float32))
    b = tf.Variable(tf.random_normal([num_class],dtype=tf.float32))
    z = tf.matmul(input, w) + b
    return z

image_size=28*28
num_class = 10

no_train_image = len(y_train)
# print no_train_image
y_train_hot = np.zeros((no_train_image, num_class))
y_train_hot[np.arange(no_train_image), y_train] = 1
# print(y_train_hot[0])

no_test_image = len(y_test)
print no_test_image
y_test_hot = np.zeros((no_test_image, num_class))
y_test_hot[np.arange(no_test_image), y_test] = 1



x = tf.placeholder(tf.float32, [None, image_size])
y_true = tf.placeholder(tf.float32, [None, num_class])
y_true_class = tf.placeholder(tf.int64, [None])


# 2 hidden layers
logit_output_1 = dense_layer_2(x,image_size,512)
logit_output = dense_layer_2(logit_output_1,512,num_class)

a = tf.nn.softmax(logit_output)
y_pred_class = tf.argmax(a, axis=1)

# usinf tf optimizer and cross_entropy for backpropagation and finding loss
error_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logit_output,labels=y_true)
cost = tf.reduce_mean(error_cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(cost)

correct_prediction = tf.equal(y_pred_class, y_true_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

batch_size = 100


def optimize(num_iterations):
	for i in range(num_iterations):
		
		random_indexes = np.random.randint(low=0,high=no_train_image,size=batch_size)
		x_batch = x_train[random_indexes]
		x_batch =np.reshape(x_batch,(batch_size,image_size))
		y_true_batch = y_train_hot[random_indexes]
		
		feed_dict_train = {x: x_batch,y_true: y_true_batch}

		sess.run(optimizer, feed_dict=feed_dict_train)

	x_test_batch =np.reshape(x_test,(no_test_image,image_size))
	feed_dict_test = {x: x_test_batch,y_true: y_test_hot,y_true_class:y_test}
	print 'accuracy = ',sess.run(accuracy, feed_dict=feed_dict_test)


def print_confusion_matrix():
	cls_true = y_test
	
	x_test_batch =np.reshape(x_test,(no_test_image,image_size))
	feed_dict_test = {x: x_test_batch,y_true: y_test_hot,y_true_class:y_test}
	
	cls_pred = sess.run(y_pred_class, feed_dict=feed_dict_test)
	cm = confusion_matrix(y_true=cls_true,y_pred=cls_pred)
	print(cm)
	recall=[]
	precision=[]
	recall_val = 0
	for i in range(len(cm)):
		print 'row_sum= ', cm[i].sum()
		num = cm[i][i]
		row_sum=cm[i].sum()
		recall_val = (1.0*num/row_sum);
		recall.append(recall_val);
		precision_val = (1.0*cm[i][i]/cm[:,i].sum());
		precision.append(precision_val);


	print recall
	print precision
	f_score=[]
	for i in range(len(recall)):
		val = 2.0 * recall[i] * precision[i]
		val /= (precision[i]+recall[i])
		f_score.append(val)

	print f_score

optimize(num_iterations=100)
# print_confusion_matrix()


