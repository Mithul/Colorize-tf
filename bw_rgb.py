# IPython log file

# get_ipython().magic(u'logstart bw_rgb.py')
import tensorflow as tf

with tf.device('/gpu:0'):
	x_raw = tf.placeholder(tf.float32, shape=(None, 64*3, 64*2, 3), name="raw_image")              
	x_bw = tf.image.rgb_to_grayscale(x_raw, name="bw_image")                                  
	def max_pool_2x2(x):                                                                
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')                
	def conv2d(x, W):                                                                   
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')            
	with tf.variable_scope("layer1") as scope:                                                                       
		W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16]), name="conv1w")
		tf.histogram_summary("layer1_w", W_conv1)
		b_conv1 = tf.Variable(tf.zeros([16]), name="conv1b")                                            
		h_conv1 = tf.nn.sigmoid(conv2d(x_bw, W_conv1) + b_conv1, name="conv1")                            
		h_pool1 = max_pool_2x2(h_conv1)  
	with tf.variable_scope("layer2") as scope:                                                                       
		W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32]), name="conv2w")                                  
		tf.histogram_summary("layer2_w", W_conv2)                                   
		b_conv2 = tf.Variable(tf.zeros([32]), name="conv2b")                                            
		h_conv2 = tf.nn.sigmoid(conv2d(h_pool1, W_conv2) + b_conv2, name="conv2")                         
		h_pool2 = max_pool_2x2(h_conv2)                                                  
	# with tf.variable_scope("layer3") as scope:                                                                       
	# 	W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 32, 4]),name="conv3")
	#     tf.histogram_summary("layer3_w", W_conv3)                                   
	# 	b_conv3 = tf.Variable(tf.zeros([4]),name="conv3")
	# 	h_conv3 = tf.nn.sigmoid(conv2d(h_pool2, W_conv3) + b_conv3,name="conv3")
	# 	h_pool3 = max_pool_2x2(h_conv3)
with tf.device('/cpu:0'):
	import cv2
	import numpy as np
	l=cv2.imread('/home/mithul/Pictures/ctfPage.png')
	ip = tf.reshape(tf.image.resize_images(np.asarray([l]),64*3,64*2),(-1,64*3,64*2,3))/255
	sess = tf.InteractiveSession()
	train_writer = tf.train.SummaryWriter('./train',sess.graph)
	with tf.variable_scope("final") as scope:                                                                       
		h_pool3_flat = tf.reshape(h_pool2, [-1, 48*32*4],name="pool3_flat")
		w_d = tf.Variable(tf.truncated_normal([48*32*4,64*64*3]),name="w_f")
		tf.histogram_summary("layerf_w", w_d)                                   
		b_d = tf.Variable(tf.zeros([64*64*3]),name="b_f")
		h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool3_flat, w_d) + b_d,name="dense")
	y = tf.image.resize_images(x_raw,64,64)
	y_h = tf.reshape(h_fc1,[-1,64,64,3], name="hypo")
	# cross_entropy = tf.minimum(tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_h + 1e-50), reduction_indices=[1])),-1000)
	cross_entropy = tf.reduce_mean( tf.pow(y - y_h,2) )
	tf.scalar_summary("loss",-tf.reduce_sum(y * tf.log(y_h + 1e-50)))
	optimizer = tf.train.AdamOptimizer(0.1)
	tvars = tf.trainable_variables()
	cost = cross_entropy
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 10.0)
	train_step = optimizer.apply_gradients(zip(grads, tvars))
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	merged = tf.merge_all_summaries()
	import matplotlib.pyplot as pyplt
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state('model/')
	step = 0
	if ckpt and ckpt.model_checkpoint_path:
		print("Checkpoint Found ")
		saver.restore(sess, ckpt.model_checkpoint_path)

	def run():
		global step
		for i in xrange(1000):
			y, _, loss = sess.run([y_h, train_step, cross_entropy], {x_raw: ip.eval()})
			if i%10==0:
				print("Loss : %f , step %d "%(loss, i))
		saver.save(sess, 'model/' + 'model.ckpt', global_step=step+1)
		step=step+1
		if i%10==0:
			cv2.imshow('n', y[0])

		if i%10==0:
			train_writer.add_summary(summary, i)

sess.run(train_step, {x_raw: ip.eval()})
