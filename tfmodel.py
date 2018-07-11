import boto3
import os
import spectrogram as S
import numpy as np
import tensorflow as tf
import pickle
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected as fc
batch_size=8
n_hidden = 128
lstm_hidden = 200
epochs=5
lstm_hidden=200
#image = tf.placeholder(dtype = tf.float32,shape = [75,1,1024],name = 'image')
#images  = tf.reshape(image, [batch_size, 75, 1, 1024], name='image')
label_spectrogram=tf.placeholder("float", [batch_size,298,257,2])
audio_tensor = tf.placeholder("float", [batch_size,298,257,2])

audio_kernel1 = tf.Variable(tf.random_normal([1,7,2,96]), dtype=tf.float32, name='audio_kernel1')
audio_kernel2 = tf.Variable(tf.random_normal([7,1,96,96]), dtype=tf.float32, name='audio_kernel2')
audio_kernel3 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel3')
audio_kernel4 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel4')
audio_kernel5 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel5')
audio_kernel6 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel6')
audio_kernel7 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel7')
audio_kernel8 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel8')
audio_kernel9 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel9')
audio_kernel10 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel10')
audio_kernel11 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel11')
audio_kernel12 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel12')
audio_kernel13 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel13')
audio_kernel14 = tf.Variable(tf.random_normal([5,5,96,96]), dtype=tf.float32, name='audio_kernel14')
audio_kernel15 = tf.Variable(tf.random_normal([1,1,96,8]), dtype=tf.float32, name='audio_kernel15')   

audio_conv1 = tf.nn.convolution(input=audio_tensor, filter=audio_kernel1, padding='SAME', strides=None, dilation_rate=None, name='audio_conv1', data_format=None)
r1 = tf.nn.relu(audio_conv1, name='r1')
audio_conv2 = tf.nn.convolution(input=r1, filter=audio_kernel2, padding='SAME', strides=None, dilation_rate=None, name='audio_conv2', data_format=None)
r2 = tf.nn.relu(audio_conv2, name='r2')
audio_conv3 = tf.nn.convolution(input=r2, filter=audio_kernel3, padding='SAME', strides=None, dilation_rate=None, name='audio_conv3', data_format=None)
r3 = tf.nn.relu(audio_conv3, name='r3')
audio_conv4 = tf.nn.convolution(input=r3, filter=audio_kernel4, padding='SAME', strides=None, dilation_rate=[2,1], name='audio_conv4', data_format=None)
r4 = tf.nn.relu(audio_conv4, name='r4')
audio_conv5 = tf.nn.convolution(input=r4, filter=audio_kernel5, padding='SAME', strides=None, dilation_rate=[4,1], name='audio_conv5', data_format=None)
r5 = tf.nn.relu(audio_conv5, name='r5')
audio_conv6 = tf.nn.convolution(input=r5, filter=audio_kernel6, padding='SAME', strides=None, dilation_rate=[8,1], name='audio_conv6', data_format=None)
r6 = tf.nn.relu(audio_conv6, name='r6')
audio_conv7 = tf.nn.convolution(input=r6, filter=audio_kernel7, padding='SAME', strides=None, dilation_rate=[16,1], name='audio_conv7', data_format=None)
r7 = tf.nn.relu(audio_conv7, name='r7')
audio_conv8 = tf.nn.convolution(input=r7, filter=audio_kernel8, padding='SAME', strides=None, dilation_rate=[32,1], name='audio_conv8', data_format=None)
r8 = tf.nn.relu(audio_conv8, name='r8')
audio_conv9 = tf.nn.convolution(input=r8, filter=audio_kernel9, padding='SAME', strides=None, dilation_rate=None, name='audio_conv9', data_format=None)
r9 = tf.nn.relu(audio_conv9, name='r9')
audio_conv10 = tf.nn.convolution(input=r9, filter=audio_kernel10, padding='SAME', strides=None, dilation_rate=[2,2], name='audio_conv10', data_format=None)
r10 = tf.nn.relu(audio_conv10, name='r10')
audio_conv11 = tf.nn.convolution(input=r10, filter=audio_kernel11, padding='SAME', strides=None, dilation_rate=[4,4], name='audio_conv11', data_format=None)
r11 = tf.nn.relu(audio_conv11, name='r11')
audio_conv12 = tf.nn.convolution(input=r11, filter=audio_kernel12, padding='SAME', strides=None, dilation_rate=[8,8], name='audio_conv12', data_format=None)
r12 = tf.nn.relu(audio_conv12, name='r12')
audio_conv13 = tf.nn.convolution(input=r12, filter=audio_kernel13, padding='SAME', strides=None, dilation_rate=[16,16], name='audio_conv13', data_format=None)
r13 = tf.nn.relu(audio_conv13, name='r13')
audio_conv14 = tf.nn.convolution(input=r13, filter=audio_kernel14, padding='SAME', strides=None, dilation_rate=[32,32], name='audio_conv14', data_format=None)
r14 = tf.nn.relu(audio_conv14, name='r14')
audio_conv15 = tf.nn.convolution(input=r14, filter=audio_kernel15, padding='SAME', strides=None, dilation_rate=None, name='audio_conv15', data_format=None)
final_signal = tf.nn.relu(audio_conv15, name='final')

final_signal = tf.layers.batch_normalization(final_signal, axis=-1, momentum=0.99, epsilon=0.001)

'''kernel1 = tf.Variable(tf.random_normal([7,1,1024,256]), dtype=tf.float32, name='kernel1')
res1 = tf.nn.convolution(images, kernel1,padding = "SAME",strides = [1,1],dilation_rate = [1,1])
conv1 = tf.nn.relu(res1, name='conv1')
kernel2 = tf.Variable(tf.random_normal([5,1,256,256]), dtype=tf.float32, name='kernel2')
res2 = tf.nn.convolution(conv1,kernel2,padding = "SAME",strides = [1,1],dilation_rate = [1,1])
conv2 = tf.nn.relu(res2, name='conv2')
kernel3 = tf.Variable(tf.random_normal([5,1,256,256]), dtype=tf.float32, name='kernel3')
res3 = tf.nn.convolution(conv2,kernel3,padding = "SAME",strides = [1,1],dilation_rate = [2,1])
conv3 = tf.nn.relu(res3, name='conv3')
kernel4 = tf.Variable(tf.random_normal([5,1,256,256]), dtype=tf.float32, name='kernel4')
res4 = tf.nn.convolution(conv3,kernel4,padding = "SAME",strides = [1,1],dilation_rate = [4,1])
conv4 = tf.nn.relu(res4, name='conv4')
kernel5 = tf.Variable(tf.random_normal([5,1,256,256]), dtype=tf.float32, name='kernel5')
res5 = tf.nn.convolution(conv4,kernel5,padding = "SAME",strides = [1,1],dilation_rate = [8,1])
conv5 = tf.nn.relu(res5, name='conv5')
reshape_conv5 = tf.reshape(conv5,[1,256,1,75],name='reshape_conv5')
kernel6 = tf.Variable(tf.random_normal([5,1,75,298]), dtype=tf.float32, name='kernel6')
res6 = tf.nn.convolution(reshape_conv5,kernel6,padding = "SAME",strides = [1,1],dilation_rate = [16,1])
conv6 = tf.nn.relu(res6, name='conv6')

final_audio = tf.reshape(final_signal, [batch_size,298,1,2056], name='final_reshaped_audio_signal')
final_visual = tf.reshape(conv6, [batch_size,298,1,256], name='final_visual')

final_input = tf.concat([final_audio,final_visual], 3)
'''
final_input = tf.reshape(final_signal, [batch_size, 298, 2056])
unstack_input = tf.unstack(final_input, 298,1)

lstm_fw_cell = rnn.BasicLSTMCell(lstm_hidden, forget_bias=1.0)
lstm_bw_cell = rnn.BasicLSTMCell(lstm_hidden, forget_bias=1.0)

outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, unstack_input, dtype=tf.float32)

fc1_output = fc(outputs, 600, tf.nn.relu)
fc2_output = fc(fc1_output, 600, tf.nn.relu)
fc3_output = fc(fc2_output, 257*2, tf.nn.sigmoid)
print (fc3_output.shape,"fc3output")
complex_mask = tf.reshape(fc3_output, [298, batch_size, 257,2])
complex_mask = tf.reshape(fc3_output, [2,298, batch_size, 257])
print (complex_mask,"complexmask")
complex_mask_result = tf.complex(complex_mask[0], complex_mask[1])
print (complex_mask_result,"complex_mask_result")
audio_tensor1=tf.reshape(audio_tensor,[2,298,257,batch_size])
complex_audio_tensor = tf.complex(audio_tensor1[0], audio_tensor1[1])
label_tensor1=tf.reshape(label_spectrogram,[2,298,257,batch_size])
complex_label_tensor = tf.complex(label_tensor1[0], label_tensor1[1])
complex_mask_result=tf.reshape(complex_mask_result,[298,257,batch_size])
spectrogram_result = complex_mask_result * complex_audio_tensor
print (spectrogram_result,label_spectrogram)
loss_op = tf.losses.mean_squared_error(complex_label_tensor, spectrogram_result)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss_op)
    #--------------------------------------------------------------------#    
def loadbatches():
    filename = "truedata"
    outfile = open(filename,'rb')    
    a=pickle.load(outfile)
    outfile.close()
    flename = "labeldata"
    otfile = open(flename,'rb')
    b=pickle.load(otfile)
    otfile.close()
    k=b.shape[0]
    b=np.reshape(b,[1, k , 298, 257, 2])
    a=np.reshape(a,[1, k , 298, 257, 2])
    x=np.concatenate((a,b),axis=0)
    print (x.shape)
    return x
batches=loadbatches()   
    #--------------------------------------------------------------------# 
print ("before coming in")
with tf.Session() as sess:
    for _ in range (epochs):
        #np.random.shuffle(batches) #randomising batches
        '''for batch in batches[0:batch_size]:'''
        print ("inbatch")
        sess.run(tf.global_variables_initializer())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #i=batch[0]
        print(batches.shape)
        np.random.shuffle(batches[0])
        np.random.shuffle(batches[1])
        print (batches[0])
        split_channel=(batches[0])[0:batch_size]
        lab_chanel=(batches[1])[0:batch_size]
        with tf.control_dependencies(update_ops):
            print ("imin")
            predout,opp,loss = sess.run([spectrogram_result,train_op,loss_op], feed_dict={ audio_tensor:split_channel,label_spectrogram:lab_chanel}) #'''image:i,'''
            print ("imout")