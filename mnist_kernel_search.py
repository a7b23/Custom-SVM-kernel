import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
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
		# operands.append(np.clip(decodeUnit(individual[i*5:(i+1)*5], x, y, operands),-10.0,10.0))
		operands.append(decodeUnit(individual[i*5:(i+1)*5], x, y, operands))
	return norm1(operands[-1], 1)



inp_emb_size = 64
hidden_size = 128

class PG_Network():
	def __init__(self,scope):
		with tf.variable_scope(scope):
			
			
			total_inps = 1 + len(binaryOps) + numUnits + len(unaryOps)
			time_emb = tf.get_variable('emb_matrix', shape = [total_inps, inp_emb_size])
			embs = time_emb
			
			weights_init = tf.contrib.layers.xavier_initializer() 

			
			operands_outs = [tf.get_variable('operand_out_'+str(i+1), shape = [hidden_size, i+len(operators)], initializer = weights_init) for i in range(numUnits)]
			unary_op_out = tf.get_variable('unary_out',shape = [hidden_size, len(unaryOps)],initializer = weights_init)
			binary_op_out = tf.get_variable('binary_out',shape = [hidden_size, len(binaryOps)],initializer = weights_init)

			self.inputs = tf.placeholder(shape=[1],dtype=tf.int32)
			#gru_cell = tf.contrib.rnn.GRUCell(hidden_size)

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
					self.logits.append(output_logits)   
					self.probs.append(output_prob)
					self.action_out.append(net[0])

			self.actions = tf.placeholder(shape = [None], dtype = tf.int32)
			self.baseline = tf.placeholder(shape = [], dtype = tf.float32)
			self.rewards = tf.placeholder(shape = [], dtype = tf.float32)
			self.lr = tf.placeholder(shape=[],dtype = tf.float32)
			losses = 0.0

			def neg_entropy_loss(probs) :
				
				return tf.reduce_sum(probs * tf.log(probs + 1e-5))

			for i in range(numUnits * 5) :
				losses += tf.reduce_mean(
					tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits[i], labels = self.actions[i:i+1]))
				

			losses = losses*(self.rewards - self.baseline)
			
			for i in range(numUnits * 5) :
				if i > 0 :
					losses += 0.0025 * neg_entropy_loss(self.probs[i])
			
			self.losses = losses    

			optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
			self.train_step = optimizer.minimize(self.losses)



train_file = 'data/mnist_train_data.txt'
validation_file = 'data/mnist_validation_data.txt'
test_file = 'data/mnist_test_data.txt'

# clf = svm.SVC(kernel = 'linear')
clf = svm.SVC(kernel="precomputed")

with open(train_file,'rb') as f :
	data = pickle.load(f, encoding = 'latin1')
	train_data = data['data'][0:1000]
	train_label = data['label'][0:1000]

with open(validation_file,'rb') as f :
	data = pickle.load(f, encoding = 'latin1')
	validation_data = data['data'][0:500]
	validation_label = data['label'][0:500]

with open(test_file,'rb') as f :
	data = pickle.load(f, encoding = 'latin1')
	test_data = data['data']
	test_label = data['label']

def get_meshgrid(a1, a2) :

	x = np.arange(len(a2))
	y = np.arange(len(a1))
	xx,yy = np.meshgrid(x, y)
	yy = np.reshape(yy, [-1])
	xx = np.reshape(xx, [-1])
	return a1[yy], a2[xx]

train_train_x, train_train_y = get_meshgrid(train_data, train_data)
validation_train_x, validation_train_y = get_meshgrid(validation_data, train_data)

print(train_train_x.shape)
print(train_train_y.shape)
print(validation_train_x.shape)
print(validation_train_y.shape)

def CustomKernelGramMatrix(X1, X2, individual, l1, l2) :

	gram_matrix = decode(individual, X1, X2)
	gram_matrix = np.reshape(gram_matrix, [l1, l2])

	return gram_matrix		

def compute_reward(individual) :
	
	try :	
		k_matrix = CustomKernelGramMatrix(train_train_x, train_train_y, individual, len(train_data), len(train_data))
		
		clf.fit(k_matrix, train_label)
		
		predictions = clf.predict(CustomKernelGramMatrix(validation_train_x, validation_train_y, individual, len(validation_data),
									len(train_data)))
		return np.mean(predictions == validation_label)

	except Exception as e :
		print(str(e))
		print(individual)
		saver.save(sess, 'models/error_model')
		sys.exit()

model = PG_Network('main')
config=tf.ConfigProto(log_device_placement=False,inter_op_parallelism_threads=12)
sess = tf.Session(config = config)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

def unityTest() :
	
	print('inside the test function')
	st = datetime.datetime.now()
	for k in range(10) :
		a = []	
		for i in range(numUnits) :
			for j in range(5) :
				if j == 0 or j == 2 :
					num = np.random.choice(len(operators) + i)
				elif j == 1 or j == 3 :
					num = np.random.choice(len(unaryOps))
				else :
					num = np.random.choice(len(binaryOps))
				if i > 0 :
					if j == 0 :
						num = len(operators)

				a.append(num)

		compute_reward(a)
		print('test succesful for iteration %d, time taken is %s'%(k, datetime.datetime.now() - st))
		st = datetime.datetime.now()

#unityTest()					
iterations = 100000
cur_baseline = None
baseline_rate = 0.95
start_lr = 0.0002
lr_decay_rate = 0.95
lr_decay_steps = 2500

episode_rewards = []
losses = []
lr = start_lr
t_ld = datetime.datetime.now()

for i in range(iterations) :

	a = sess.run(model.action_out, feed_dict = {model.inputs : np.array([len(binaryOps)])})

	episode_reward = compute_reward(a)
	episode_rewards.append(episode_reward)
	if cur_baseline is None :
		cur_baseline = episode_reward
	else :
		cur_baseline = cur_baseline*baseline_rate + episode_reward*(1 - baseline_rate)

	if i >0 and i%lr_decay_steps == 0 :
		lr *= lr_decay_rate
	_,loss = sess.run([model.train_step, model.losses], feed_dict = {model.inputs : np.array([len(binaryOps)]), model.actions : np.array(a),
					model.rewards : episode_reward, model.baseline : cur_baseline, model.lr : lr})
	losses.append(loss)
	if i%5 == 0 :
		print('%s : after %d steps the mean reward is %g and loss is %g'%(datetime.datetime.now(),i, np.mean(np.array(episode_rewards[-10:]))
			, np.mean(np.array(losses[-10:]))))

	if i%500 == 0 and i >0:
		saver.save(sess,'models/model_'+str(i))
		print('model saved')						
	

