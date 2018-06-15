import sys
import numpy as np
import tensorflow as tf
import math
import skimage
import six
from sklearn.metrics import confusion_matrix

config = tf.ConfigProto()#log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.45
sess = tf.InteractiveSession("",config=config)

def unpickle(file):
    import pickle
    fo=open(file,'rb')
    dict=pickle.load(fo)
    fo.close()
    return dict

def weight_variable(shape, n):
    initial = tf.truncated_normal(shape, stddev=2/math.sqrt(n)) #var(w) = 2/n
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# image load
t1 = unpickle("train1.pkl")
t2 = unpickle("test.pkl")

train_images = t1["data"]
train_images_ =t1["labels"]

test_images = t2["data"]
test_images_ =t2["labels"]


train_len = len(train_images_)
test_len = len(test_images_)



result_arr=[]
filter_size=4
batch_size=5
batch_num=400
epoch_num=60


sess = tf.InteractiveSession()

x1 = tf.placeholder(tf.float32, [None,9919])
y1_ = tf.placeholder(tf.float32, [None,5])




# first layer input_size:32x32, output_size:16x16 kernel_size:5x5 input_chanel:3 output_chanel:32
W_conv11 = weight_variable([filter_size, filter_size, 1, 32], filter_size*filter_size*1)
b_conv11 = bias_variable([32])
x_image1 = tf.reshape(x1, [-1, 91, 109, 1])
h_conv11 = tf.nn.relu(conv2d(x_image1, W_conv11) + b_conv11)
h_pool11 = max_pool_2x2(h_conv11)
	

# second layer input_size:16x16, output_size:8x8 kernel_size:5x5 input_cahnel:32 output_chanel:64
W_conv21 = weight_variable([filter_size,filter_size ,32, 64], filter_size*filter_size*32)
b_conv21 = bias_variable([64])
h_conv21 = tf.nn.relu(conv2d(h_pool11, W_conv21) + b_conv21)
h_pool21 = max_pool_2x2(h_conv21)


# 3th layer input_size:16x16, output_size:8x8 kernel_size:5x5 input_cahnel:32 output_chanel:64
W_conv31 = weight_variable([filter_size,filter_size, 64, 128], filter_size*filter_size*64)
b_conv31 = bias_variable([128])
h_conv31 = tf.nn.relu(conv2d(h_pool21, W_conv31) + b_conv31)
h_pool31 = max_pool_2x2(h_conv31)



# fully-connected layer
W_fc11 = weight_variable([9*24*128, 1024], 9*24*128) # [h*w*in_c, out_c]
b_fc11 = bias_variable([1024])
h_pool2_flat1 = tf.reshape(h_pool31, [-1, 9*24*128])
h_fc11 = tf.nn.relu(tf.matmul(h_pool2_flat1, W_fc11) + b_fc11)



# dropout
keep_prob1 = tf.placeholder(tf.float32)
h_fc1_drop1 = tf.nn.dropout(h_fc11, keep_prob1)

# readout layer
W_fc21 = weight_variable([1024, 5],1024)
b_fc21 = bias_variable([6])
y_conv1 = tf.nn.softmax(tf.matmul(h_fc1_drop1, W_fc21) + b_fc21)

	
# train 
a1 = np.zeros((train_len,5))
a1[np.arange(train_len), train_images_] = 1

	

# test
a1_ = np.zeros((test_len,5))
a1_[np.arange(test_len), test_images_] = 1

# train and evaluate!!
cross_entropy1 = tf.reduce_mean(-tf.reduce_sum(y1_ * tf.log(y_conv1), reduction_indices=[1]))
train_step1 = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy1)

pred=tf.argmax(y_conv1, 1)
real=tf.argmax(y1_, 1)
correct_prediction1 = tf.equal(pred, real)
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
sess.run(tf.global_variables_initializer())
#saver = tf.train.Saver()
#saver.restore(sess,"./save_pa/F5E50_double.ckpt")

#save model


print("=========SET_Parameter==========")
print('Train dataset: ',train_images.shape)
print( 'test dataset: ',test_images.shape)
print ("filter size : ",filter_size)
print ("batch size : ",batch_size)
print ("")
print("===========Operate model============")



for epoch in range(0,epoch_num):
	print("%d epoch"%(epoch+1))
		
	
	for i in range(batch_num):

		batch = train_images[i*batch_size:i*batch_size+batch_size]
		l_batch = a1[i*batch_size:i*batch_size+batch_size]
		batch = (batch-128)/128.0
		if i%(batch_num-1) == 0:
			if i!=0:
			    train_accuracy1 = accuracy1.eval(feed_dict={x1:batch, y1_: l_batch, keep_prob1: 1.0})
			    print("step %d, training accuracy1 %g"%(i,train_accuracy1))

		train_step1.run(feed_dict={x1:batch, y1_:l_batch, keep_prob1:0.5})
		
		
	if  (epoch+1)%5==0:
		
		#save_path = saver.save(sess, "save_pa/F"+str(5)+"E"+str(epoch+1)+"_double.ckpt")
		#save_path = saver.save(sess,"/home/cvmlmoon/Desktop/model1.ckpt")
		last_accuracy=0
		
		real_fl=[]
		pred_fl=[]
		for k in range(15):


			batch = test_images[k*21:k*21+21]
			batch = (batch-128)/128.0
			l_batch = a1_[k*21:k*21+21]

			test_accuracy = accuracy1.eval(feed_dict={x1: batch, y1_: l_batch, keep_prob1:1.0})
			pred_l=sess.run(pred,feed_dict={x1:batch, y1_:l_batch, keep_prob1:1.0})
			real_l=sess.run(real,feed_dict={x1:batch, y1_:l_batch, keep_prob1:1.0})
			for c in range(0,len(pred_l)):
				pred_fl.append(pred_l[c])
				real_fl.append(real_l[c])

			last_accuracy = last_accuracy+test_accuracy
		r=confusion_matrix(real_fl,pred_fl)
		print (r)
		bun0=0
		bun1=0
		bun2=0
		bun3=0
		bun4=0
		bun5=0
		bun6=0
		for in1 in range(0,5):
			for in2 in range(0,5):
				if in1==0:
					bun0=bun0+r[in2][0]
				if in1==1:
					bun1=bun1+r[in2][1]
				if in1==2:
					bun2=bun2+r[in2][2]
				if in1==3:
					bun3=bun3+r[in2][3]
				if in1==4:
					bun4=bun4+r[in2][4]
				
				
		if bun0 !=0 and bun1 !=0 and bun2 !=0 and bun3 !=0 and bun4 !=0  :

			print ("===========result==========")
			print ("epoch: ",epoch+1)
			print ("filter size: ",filter_size)
			print ("test accuracy = ",last_accuracy/15)
			precision0=float(r[0][0])/bun0
			precision1=float(r[1][1])/bun1
			precision2=float(r[2][2])/bun2
			precision3=float(r[3][3])/bun3
			precision4=float(r[4][4])/bun4
			
			av_precision=(precision0+precision1+precision2+precision3+precision4)/5.0
			print ("precision0 : ",precision0)
			print ("precision1 : ",precision1)
			print ("precision2 : ",precision2)
			print ("precision3 : ",precision3)
			print ("precision4 : ",precision4)
			
			

			print (" average Precision : ",av_precision)
			#print "save_path : ",save_path
		else : 
			print ("zeroDivisionError ")
			print ("===========result==========")
			print ("epoch: ",epoch+1)
			print ("filter size: ",filter_size)
			print ("test accuracy = ",last_accuracy/15)
			if bun0 != 0:
				precision0=float(r[0][0])/bun0
			else : 
				precision0 =0
			if bun1 != 0:
				precision1=float(r[1][1])/bun1
			else : 
				precision1 =0
			if bun2 != 0:
				precision2=float(r[2][2])/bun2
			else : 
				precision2 =0
			if bun3 != 0:
				precision3=float(r[3][3])/bun3
			else : 
				precision3 =0
			if bun4 != 0:
				precision4=float(r[4][4])/bun4
			else : 
				precision4 =0
			
			


			av_precision=(precision0+precision1+precision2+precision3+precision4)/5.0
			print ("precision0 : ",precision0)
			print ("precision1 : ",precision1)
			print ("precision2 : ",precision2)
			print ("precision3 : ",precision3)
			print ("precision4 : ",precision4)
			
			
			print (" average Precision : ",av_precision)
		

			 



