import tensorflow as tf

class TrajBlock():
	def __init__(self,layer_name, filter_size,num_hidden,keep_prob):
		self.layer_name=layer_name
		self.filter_size=filter_size
		self.num_hidden=num_hidden
		self.keep_prob = keep_prob

	def __call__(self, h, reuse=False):
		with tf.variable_scope(self.layer_name, reuse=False):
			num_hidden = self.num_hidden
			out=[]
			filter_size=self.filter_size
			#h0 = h
			h0 = tf.layers.conv2d(h,num_hidden[0],1, padding='same',activation=tf.nn.leaky_relu,   
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='h0')
			traj1 = tf.layers.conv2d(h,num_hidden[0],filter_size, padding='same',activation=tf.nn.leaky_relu,  
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='layer1')
			traj1 = tf.nn.dropout(traj1, self.keep_prob)
			h1 = tf.layers.conv2d(traj1,num_hidden[1],1, padding='same',activation=tf.nn.leaky_relu,   
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					 name='h1')
			traj2 = tf.layers.conv2d(traj1,num_hidden[1],filter_size, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='layer2')
			traj2 = tf.nn.dropout(traj2, self.keep_prob)
			h2 = tf.layers.conv2d(traj2,num_hidden[2],1, padding='same',activation=tf.nn.leaky_relu,   
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					 name='h2')
			traj3 = tf.layers.conv2d(traj2, num_hidden[2], filter_size, padding='same',activation=tf.nn.leaky_relu,
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='layer3') 
			traj3 = tf.nn.dropout(traj3, self.keep_prob)           
			traj4 = tf.layers.conv2d(traj3+h2, num_hidden[3], filter_size, padding='same',activation=tf.nn.leaky_relu, 
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='layer4')
			traj4 = tf.nn.dropout(traj4, self.keep_prob)
			traj5 = tf.layers.conv2d(traj4+h1, num_hidden[4] ,filter_size, padding='same',activation=tf.nn.leaky_relu, 
					kernel_initializer=tf.contrib.layers.xavier_initializer(),
					name='layer5')
			traj5 = tf.nn.dropout(traj5, self.keep_prob)
			# loss
			out = traj5+h0
			return out

