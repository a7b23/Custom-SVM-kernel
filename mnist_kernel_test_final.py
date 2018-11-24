import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import scipy.signal
import random
import os
import sys
import pickle

from sklearn import svm
import datetime


def unity(x) :
	return x

def negation(x) :
	return -x

def absolute(x) :
	return np.abs(x)

def power(x,k) :
	return np.power(x,k)

def sin(x) :
	return np.sin(x)

def cos(x) :
	return np.cos(x)

def tan(x) :
	return np.tan(x)

def exp(x) :
	return np.exp(x)

def log(x) :
	return np.log(x)

def sqrt(x) :
	return np.sqrt(np.abs(x))

def sinh(x) :
	return np.sinh(x)

def cosh(x) :
	return np.cosh(x)

def tanh(x) :
	return np.tanh(x)

def inverse_sinh(x) :
	return np.arcsinh(x)

def inverse_tan(x) :
	return np.arctan(x)             

def sinc(x) :
	return np.sinc(x)

def sigmoid(x) :
	return 1.0/(1.0 + np.exp(-x))

def maximum(x,y) :
	return np.maximum(x,y)

def minimum(x,y) :
	return np.minimum(x,y)

def erf(x) :
	# return math.erf(x)
	return scipy.special.erf(x)

def summation(x,y) :
	return np.add(x,y)

def difference(x,y) :
	return np.subtract(x,y)

def multiply(x,y) :
	return np.multiply(x,y)

def divide(x,y) :
	return np.divide(x,y)           

def norm(x, k) :
	if np.array(x).shape[-1] == 1 :
		return x
	return np.linalg.norm(x, ord = k, axis = -1, keepdims = True)*gamma(x, 1)

def gamma(x, y) :
	# return 1.0/float(len(x))
	return 1.0/float(train_data.shape[1])

def squared_diff(x, y) :
	return absolute(difference(x**2, y**2))
	# return absolute(x**2 - y**2)

def squared_sum(x, y) :
	return summation(x**2, y**2)
	# return x**2 + y**2

def dot_prod(x, y) :
	if np.array(x).shape[-1] == 1 and np.array(y).shape[-1] == 1 :
		return multiply(x, y)

	return np.sum(multiply(x, y), axis = -1, keepdims = True)*gamma(x, y)

def norm1(x, k) :
	if np.array(x).shape[-1] == 1 :
		return x
	return np.linalg.norm(x, ord = k, axis = -1)

numUnits = 2
eps = 1e-5

operators = [lambda x, y : absolute(difference(x, y)), summation, squared_diff,
			lambda x, y : norm(difference(x, y), 1), lambda x, y : norm(difference(x, y), 2), dot_prod, 
			multiply]

unaryOps = [unity, negation, absolute, lambda x : power(x,2), lambda x : power(x,3), sqrt, exp, sin, cos,
			sinh, cosh, tanh, lambda x : maximum(x,0), lambda x : minimum(x,0), sigmoid,
			lambda x : log(1 + exp(x)), lambda x : norm(x, 1), lambda x : norm(x, 2)]

binaryOps = [summation, multiply, difference, 
			 maximum, minimum,dot_prod, 
			 lambda x,y : exp(-1 * absolute(difference(x,y))),
			 lambda x,y : x]

def decodeUnit(unit, x, y, operands) :
	if unit[0] < len(operators) :
		operand1 = operators[unit[0]](x, y)
	else :
		num = unit[0] - len(operators)
		operand1 = 	operands[num + 1]

	if unit[2] < len(operators) :
		operand2 = operators[unit[2]](x, y)
	else :
		num = unit[2] - len(operators)
		operand2 = 	operands[num + 1]

	return 	binaryOps[unit[4]](unaryOps[unit[1]](operand1), unaryOps[unit[3]](operand2))



def decode(individual, x, y) :
	operands = [1]
	for i in range(numUnits) :
		operands.append(decodeUnit(individual[i*5:(i+1)*5], x, y, operands))
	return norm1(operands[-1], 1)



inp_emb_size = 64
hidden_size = 128

class PG_Network():
	def __init__(self,scope):
		with tf.variable_scope(scope):
			
			
			total_inps = 1 + len(binaryOps) + numUnits + len(unaryOps)
			time_emb = tf.get_variable('emb_matrix', shape = [total_inps, inp_emb_size])
			# embs = [time1_emb, time2_emb, time3_emb, time2_emb, time3_emb]
			embs = time_emb
			
			weights_init = tf.contrib.layers.xavier_initializer() 

			
			operands_outs = [tf.get_variable('operand_out_'+str(i+1), shape = [hidden_size, i+len(operators)], initializer = weights_init) for i in range(numUnits)]
			unary_op_out = tf.get_variable('unary_out',shape = [hidden_size, len(unaryOps)],initializer = weights_init)
			binary_op_out = tf.get_variable('binary_out',shape = [hidden_size, len(binaryOps)],initializer = weights_init)

			self.inputs = tf.placeholder(shape=[1],dtype=tf.int32)

			net = self.inputs
			state = tf.zeros(shape = [1,hidden_size], dtype = tf.float32)
			self.action_out = []
			self.logits = []
			self.probs = []
			for i in range(numUnits) :
				for j in range(5) :
					if i == 0 and j == 0 :
						gru_cell = tf.contrib.rnn.GRUCell(hidden_size)
					else :
						gru_cell = tf.contrib.rnn.GRUCell(hidden_size,reuse = True) 
					output,state = gru_cell(tf.nn.embedding_lookup(embs,net),state)
					
					if j == 0 or j == 2 :
						output_logits = tf.matmul(output, operands_outs[i])
					elif j == 1 or j == 3 :
						output_logits = tf.matmul(output, unary_op_out)
					elif j == 4 :
						output_logits = tf.matmul(output, binary_op_out)

					output_prob = tf.nn.softmax(output_logits)

					
					a_prob = tf.multinomial(tf.log(output_prob), 1)    
						
					net = tf.reshape(tf.cast(a_prob[0][0], tf.int32),[-1])  
					self.action_out.append(net[0])


train_file = 'data/mnist_train_data.txt'
validation_file = 'data/mnist_validation_data.txt'

# clf = svm.SVC(kernel = 'linear')
clf = svm.SVC(kernel="precomputed")

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)  

with open(train_file,'rb') as f :
	data = pickle.load(f, encoding = 'latin1')
	train_data = data['data'][0:1000]
	train_label = data['label'][0:1000]

with open(validation_file,'rb') as f :
	data = pickle.load(f, encoding = 'latin1')
	validation_data = data['data'][0:500]
	validation_label = data['label'][0:500]

test_data = mnist.test.images
test_label = np.argmax(mnist.test.labels, axis = -1)

def get_meshgrid(a1, a2) :

	x = np.arange(len(a2))
	y = np.arange(len(a1))
	xx,yy = np.meshgrid(x, y)
	yy = np.reshape(yy, [-1])
	xx = np.reshape(xx, [-1])
	return a1[yy], a2[xx]

train_train_x, train_train_y = get_meshgrid(train_data, train_data)
validation_train_x, validation_train_y = get_meshgrid(validation_data, train_data)
test_train_x, test_train_y = get_meshgrid(test_data, train_data)

print(train_train_x.shape)
print(train_train_y.shape)
print(validation_train_x.shape)
print(validation_train_y.shape)

def CustomKernelGramMatrix(X1, X2, individual, l1, l2) :

	gram_matrix = decode(individual, X1, X2)
	gram_matrix = np.reshape(gram_matrix, [l1, l2])

	return gram_matrix		

def compute_reward(individual, test_train_x, test_train_y, test_label) :
	
	try :	
		test_predictions = clf.predict(CustomKernelGramMatrix(test_train_x, test_train_y, individual, 1000,
									len(train_data)))
		return np.mean(test_predictions == test_label)

	except Exception as e :
		print(str(e))
		print(individual)
		saver.save(sess, 'mnist_models/error_model')
		sys.exit()

def evaluate(individual, test_train_x, test_train_y, test_label) :
	print(individual)
	test_acc = compute_reward(individual, test_train_x, test_train_y, test_label)
	print('the test acc is ', test_acc)
	return test_acc

model = PG_Network('main')
config=tf.ConfigProto(log_device_placement=False,inter_op_parallelism_threads=12)
config.gpu_options.allow_growth = True

sess = tf.Session(config = config)
saver = tf.train.Saver()

kernels = ['linear', 'rbf', 'sigmoid']
for kernel in kernels :
	print('doing evaluations for the kernel ', kernel)
	clf1 = svm.SVC(kernel = kernel)
	clf1.fit(train_data, train_label)
	predictions = clf1.predict(validation_data)
	print(np.mean(predictions == validation_label), 'normal validation acc')
	predictions = clf1.predict(test_data)
	print(np.mean(predictions == test_label),'normal test acc')


saver.restore(sess,'models/model_9000')


a = [6, 6, 2, 7, 0, 6, 7, 5, 7, 4]
# a = sess.run(model.action_out, feed_dict = {model.inputs : np.array([len(binaryOps)])})
total_correct = 0
k_matrix = CustomKernelGramMatrix(train_train_x, train_train_y, a, len(train_data), len(train_data))
clf.fit(k_matrix, train_label)
for i in range(10) :
	new_test_data = test_data[i*1000:(i+1)*1000]
	new_test_label = test_label[i*1000:(i+1)*1000]
	test_train_x, test_train_y = get_meshgrid(new_test_data, train_data)
	acc = evaluate(a, test_train_x, test_train_y, new_test_label)
	total_correct += acc*1000
print('the final acc is ', float(total_correct)/10000)	
	

