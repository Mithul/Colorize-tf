import tensorflow as tf
# 237.870598
# 237.393567
# 237.188018
# 225.409334
height = 256
width = 256
with tf.device('/gpu:0'):
	x_raw = tf.placeholder(tf.float32, shape=(None, height, width, 3), name="raw_image")              
	x_bw = tf.image.rgb_to_grayscale(x_raw, name="bw_image")  
	y = x_bw

with tf.device('/cpu:0'):
	x_bw_u = tf.image.resize_images(x_bw, height*4, width*4)
with tf.device('/gpu:0'):

	r,g,b = tf.split(3,3,x_raw)
   	u = 0.492 *(b-y)
   	v = 0.877 *(r-y)

	# u = (128-0.168736*r -0.331364*g + 0.5*b)
	# v = 128 +.5*r - .418688*g - .081312*b

	x_uv = tf.concat(3, [u, v])                        

	def max_pool_2x2(x):                                                                
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')                
	def conv2d(x, W):                                                                   
		return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')            
	with tf.variable_scope("layer1") as scope:                                                                       
		W_conv1 = tf.Variable(tf.truncated_normal([1, 1, 1, 512]), name="conv1w")
		# tf.histogram_summary("layer1_w", W_conv1)
		b_conv1 = tf.Variable(tf.zeros([512]), name="conv1b")                                            
		h_conv1 = tf.nn.relu(conv2d(x_bw, W_conv1) + b_conv1, name="conv1")                            
	with tf.variable_scope("layer2") as scope:                                                                       
		W_conv2 = tf.Variable(tf.truncated_normal([1, 1, 512, 128]), name="conv2w")                                  
		# tf.histogram_summary("layer2_w", W_conv2)                                   
		b_conv2 = tf.Variable(tf.zeros([128]), name="conv2b")                                            
		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2, name="conv2")                         

	with tf.variable_scope("layer3") as scope:                                                                       
		W_conv3 = tf.Variable(tf.truncated_normal([1, 1, 128, 2]), name="conv3w")                                  
		# tf.histogram_summary("layer2_w", W_conv2)                                   
		b_conv3 = tf.Variable(tf.zeros([2]), name="conv2b")                                            
		h_conv3 = tf.nn.tanh(conv2d(h_conv2, W_conv3) + b_conv3, name="conv3")                

	# tt = tf.concat(3,tf.split(3,2,h_pool2))

	#upscale
	# channels = tf.split(3,16,h_pool2)
	# upscale_r = [tf.concat(2,	[tf.concat(1,channels[0:4]), tf.concat(1,channels[4:8])])],[tf.concat(2,[tf.concat(1,channels[8:12]), tf.concat(1,channels[12:16])])]
	# upscale = tf.reshape(tf.concat(3, upscale_r),[-1, 256, 256, 2])
	# y_uv = upscale
	# y_uv = tf.reshape(h_pool2, [-1, 256, 256, 2])
	# y_uv = tf.depth_to_space(h_pool2, 4)
	y_uv = h_conv3
	# y_uv = tf.clip_by_value(y_uv, -255 , 255)

	# with tf.variable_scope("layer3") as scope:                                                                       
	# 	W_conv3 = tf.Variable(tf.truncated_normal([5, 5, 32, 2]),name="conv3")
	# 	tf.histogram_summary("layer3_w", W_conv3)                                   
	# 	b_conv3 = tf.Variable(tf.zeros([2]),name="conv3")
	# 	h_conv3 = tf.nn.sigmoid(conv2d(h_pool2, W_conv3) + b_conv3,name="conv3")
	# 	h_pool3 = max_pool_2x2(h_conv3)
	sess = tf.InteractiveSession()

	
with tf.device('/gpu:0'):
	# y = tf.image.resize_images(x_raw,height, width)
	# y_h = tf.reshape(h_fc1,[-1, height, width,3], name="hypo")

	y = x_uv
	y_h = y_uv
	# y_h = tf.clip_by_value(y_h, -255 , 255)

	y_orig_uv = tf.concat(3, [x_bw, y])

	y_i, u, v =  tf.split(3,3,y_orig_uv)

	r = y_i + 1.140*v
	g = y_i - 0.395*u - 0.581*v
	b = y_i + 2.032*u
	
	# r = y_i + 1.402 * (u-128)
	# g = y_i - .34414 * (v-128) -  .71414 * (u-128)
	# b = y_i + 1.772 * (v-128)

	y_orig = tf.concat(3, [r, g, b])

	y_h_orig_uv = tf.concat(3, [x_bw, y_h])

	y_i, u, v =  tf.split(3,3,y_h_orig_uv)
	r = y_i + 1.140*v
	g = y_i - 0.395*u - 0.581*v
	b = y_i + 2.032*u

	y_h_orig = tf.concat(3, [r, g, b])

	# y = y*255
	# y_h = y_h*255

	# y_h = y_orig
	# y = x_raw


	def gkern(kernlen=21, nsig=3):
		"""Returns a 2D Gaussian kernel array."""
		import numpy as np
		import scipy.stats as st

		interval = (2*nsig+1.)/(kernlen)
		x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
		kern1d = np.diff(st.norm.cdf(x))
		kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
		kernel = kernel_raw/kernel_raw.sum()
		return kernel

	def blur_tensor(kernel, ip):
		core = tf.cast(gkern(kernel),tf.float32)
		filter = tf.concat(3,[tf.expand_dims(tf.concat(2,[tf.expand_dims(core,-1),tf.zeros([kernel,kernel,1]),tf.zeros([kernel,kernel,1])]),-1),
		tf.expand_dims(tf.concat(2,[tf.zeros([kernel,kernel,1]),tf.expand_dims(core,-1),tf.zeros([kernel,kernel,1])]),-1),
		tf.expand_dims(tf.concat(2,[tf.zeros([kernel,kernel,1]),tf.zeros([kernel,kernel,1]),tf.expand_dims(core,-1)]),-1)])
		 
		blur = tf.nn.conv2d(ip,tf.cast(filter,tf.float32), strides=[1,1,1,1], padding='SAME')
		return blur


	y_orig_ba = (y_orig_uv+blur_tensor(3,y_orig_uv)+blur_tensor(5,y_orig_uv))/3

	lr = tf.Variable(0.01)
	cross_entropy_b = tf.reduce_mean(tf.reduce_sum((y +1) * tf.log(y_h +1 + 1e-50)))
	cross_entropy = tf.reduce_mean( tf.pow(y_orig_ba - y_h_orig_uv,2) )
	# tf.scalar_summary("loss",-tf.reduce_sum(y * tf.log(y_h + 1e-50)))
	optimizer = tf.train.GradientDescentOptimizer(lr)
	tvars = tf.trainable_variables()
	cost = tf.abs(cross_entropy)
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 10.0)
	train_step = optimizer.apply_gradients(zip(grads, tvars))
	init_op = tf.initialize_all_variables()
	sess.run(init_op)
	# merged = tf.merge_all_summaries()
	import matplotlib.pyplot as pyplt	
	
with tf.device('/cpu:0'):
	import cv2
	import numpy as np



	def get_data(batch_size=5):
		from os import listdir
		from os.path import isfile, join
		mypath = "/media/mithul/Common/Photos/Dataset/"
		onlyfiles = [ join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
		i=0
		import random
		random.shuffle(onlyfiles)
		while True:
			images = []
			# onlyfiles = onlyfiles[0:5]
			for f in onlyfiles[(i)*batch_size:(i+1)*batch_size]:
				try:
				# print(f)
					images.append(np.asarray(cv2.resize(cv2.imread(f),(256,256)),dtype=np.float32)/255)
				except:
					print("Error : %s"%f)

			# with tf.variable_scope("inputs", reuse=True):
				# ip = tf.get_variable("input", (5, height, width, 3))
			# sess.run(tf.assign(ip, tf.reshape(tf.image.resize_images(np.asarray(images,dtype=np.float32),height, width),
			# (-1,height, width,3))/255.0))
			# print(np.asarray(images).shape)
			yield images
			i=i+1
			if (i+1)*5 >= len(onlyfiles):
				i=0 

	# l=cv2.imread('/home/mithul/Pictures/2016/05/15/IMG_1757_CR2_shotwell.jpg')
	
	train_writer = tf.train.SummaryWriter('./train',sess.graph)
	saver = tf.train.Saver()
	ckpt = tf.train.get_checkpoint_state('model4/')
	step = 0
	if ckpt and ckpt.model_checkpoint_path:
		print("Checkpoint Found ")
		saver.restore(sess, ckpt.model_checkpoint_path)

	sess.run(tf.assign(lr, 0.001))

	def show(ip, i=0):
		cv2.imshow('o',sess.run(tf.reshape(tf.split(2,3,tf.concat(3,[x_bw,y])[i]),[-1,256*3,256]), {x_raw: ip})[0])
		cv2.imshow('h',sess.run(tf.reshape(tf.split(2,3,tf.concat(3,[x_bw,y_h])[i]),[-1,256*3,256]), {x_raw: ip})[0])
		# cv2.imshow('n2',sess.run(tf.reshape([y_h_orig, y_orig, tf.concat(3,[x_bw]*3)[i]],[-1,256,256*3,3])[0], {x_raw: ip.eval()})[0][3])
		cv2.imshow('n2',sess.run(tf.reshape(tf.concat(2,[y_h_orig, y_orig, tf.concat(3,[x_bw]*3)]),[-1,256,256*3,3]),{x_raw: ip})[i])
		# cv2.imshow('n1',sess.run([y_orig], {x_raw: ip.eval()})[0][3])
		# cv2.imshow('n3',sess.run([x_bw], {x_raw: ip.eval()})[0][3])


	def run(batch_size=5):
		ips = get_data(batch_size)
		global step
		loss_avg = 0.0
		for i in xrange(1000):
			ip = ips.next()
			y, _, loss, loss2 = sess.run([y_h, train_step, cross_entropy, cross_entropy_b], {x_raw: ip})
			loss_avg = (loss_avg*i + loss)/((i+1))
			if i%10==0:
				print("Loss : %f , %f , %f , step %d "%(loss, loss2, loss_avg, i))
		saver.save(sess, 'model4/' + 'model.ckpt', global_step=step+1)
		step=step+1
		print(y[0])


[run() for i in xrange(100)]
# 		if i%10==0:
# 			cv2.imshow('n', y[0])

# 		if i%10==0:
# 			train_writer.add_summary(summary, i)

# sess.run(train_step, {x_raw: ip.eval()})

# r,g,b = tf.split(3,3,x_raw*255)
# u = (128-0.168736*r -0.331364*g + 0.5*b)
# v = 128 +.5*r - .418688*g - .081312*b

def blur_tensor(kernel=5):
	core = tf.cast(gkern(kernel),tf.float32)
	filter = tf.concat(3,[tf.expand_dims(tf.concat(2,[tf.expand_dims(core,-1),tf.zeros([kernel,kernel,1]),tf.zeros([kernel,kernel,1])]),-1),
	tf.expand_dims(tf.concat(2,[tf.zeros([kernel,kernel,1]),tf.expand_dims(core,-1),tf.zeros([kernel,kernel,1])]),-1),
	tf.expand_dims(tf.concat(2,[tf.zeros([kernel,kernel,1]),tf.zeros([kernel,kernel,1]),tf.expand_dims(core,-1)]),-1)])
	 
	blur = tf.nn.conv2d(np.asarray(ip),tf.cast(filter,tf.float32), strides=[1,1,1,1], padding='SAME')
	return blur