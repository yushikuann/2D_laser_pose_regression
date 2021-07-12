import os
import random
import skimage.data
from skimage import io
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm

# tf.enable_eager_execution()
label_dir = "/home/ysk/code/tensorflowcode/laser_scan_cnn_localition/datasets"
pic_dir = "/home/ysk/code/tensorflowcode/laser_scan_cnn_localition/datasets/picture"

def load_data(pic_dir, label_dir):
    train_txt = os.path.join(label_dir, "label.txt")
    labels = []
    images = []
    with open(train_txt, 'r') as f:
        all_file_name = [x.strip() for x in f.readlines()]
        for fil in all_file_name:
            img = skimage.data.imread(os.path.join(pic_dir, fil.split(";")[0]))
            img = skimage.transform.resize(img, (448, 448), mode='constant')
            images.append(img)
            label = [fil.split(";")[1:]]
            labels.append(label)
    return images, labels
images, labels = load_data(pic_dir, label_dir)

#
# def get_Batch(data, label, batch_size):
#     print(np.shape(data), np.shape(label))
#     image_placeholder = tf.placeholder(data.dtype, data.shape)
#     labels_placeholder = tf.placeholder(label.dtype, label.shape)
#     input_queue = tf.data.Dataset.from_tensor_slices((image_placeholder, labels_placeholder))
#     x_batch, y_batch = input_queue.batch(batch_size)
#     return x_batch, y_batch

image_placeholder = tf.placeholder(np.array(images).dtype, np.array(labels).shape)
labels_placeholder = tf.placeholder(np.array(images).dtype, np.array(labels).shape)
datasets = tf.data.Dataset.from_tensor_slices((image_placeholder, labels_placeholder))
iter = datasets.make_one_shot_iterator()

x,y = iter.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(tf.shape(x)),sess.run(tf.shape(y)))
# print(x)
# x, y = get_Batch(np.array(images), np.array(labels), 5)

# print(np.shape(x))
# print(np.shape(y))
# with tf.Session() as sess:
#     print("shape of outputs is:",sess.run(tf.shape(x)))
#     print("shape of outputs is:",sess.run(tf.shape(y)))


# 解决： ValueError: GraphDef cannot be larger than 2GB
# features_placeholder = tf.placeholder(features.dtype, features.shape)
# labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
# dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))

# with tf.name_scope("input"):
#     input_image = tf.placeholder("float", [None, 448,448,3],name='input_image')
#     # input_image = tf.pad(image,np.array([[0, 0], [3, 3], [3, 3], [0, 0]]))
#     y_ = tf.placeholder("float", [None, 1, 3],name='y_')

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)

# def bias_variable(shape):
#     initial = tf.constant(0.1, shape = shape)
#     return tf.Variable(initial)

# def conv2d(x, W):
#     #卷积函数实现卷积层的前向传播
#     #其中第一个参数ｘ为当前节点的矩阵，为一个４维矩阵
#     #第二个参数为卷积核的值
#     #第三个参数为不同维度的步长第一个和最后一个参数必须为１
#     #第四个参数为填充，SAME为０添加，VALLD为不添加
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
#     #返回值仍然是一个４维的tensor
#     #第一维为batch数量
#     #第二维和第三维表示卷积层的维度，由输入层和卷集核以及移动步长共同决定
#     #第四维的参数为自己定义的卷积层的深度

# def max_pool_2x2(x):
#     return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# #第一层卷积层
# #参数的前两维为卷积核的维度，第三个参数为当前层的深度，第四个为输出到层的深度
# W_conv1 = weight_variable([7, 7, 3, 64])
# b_conv1 = bias_variable([64])

# h_conv1 = tf.nn.relu(tf.nn.conv2d(input_image, W_conv1,strides=[1,2,2,1],padding="SAME") + b_conv1)
# h_pool1 = max_pool_2x2(h_conv1)   #输出变成112*112*64

# #第二层卷积层
# W_conv2 = weight_variable([3, 3, 64, 192])
# b_conv2 = bias_variable([192])

# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool2 = max_pool_2x2(h_conv2)  #输出变为56*56*192


# #第三层卷积层
# W_conv3 = weight_variable([3, 3, 192, 128])
# b_conv3 = bias_variable([128])

# h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
# h_pool3 = max_pool_2x2(h_conv3)   #输出为28*28*128


# #第四层卷积层
# W_conv4 = weight_variable([3, 3, 128, 64])
# b_conv4 = bias_variable([64])

# h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
# h_pool4 = max_pool_2x2(h_conv4)   #输出变为14*14*64


# #密集连接层1
# W_fc1 = weight_variable([14*14*64, 1024])
# b_fc1 = bias_variable([1024])

# h_pool2_flat = tf.reshape(h_pool4, [-1, 14*14*64])
# h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   #输出为96*1024

# keep_prob = tf.placeholder("float")
# #relu函数作用是求出max(h_fc1,0)
# h_fc1_drop1 = tf.nn.dropout(h_fc1, keep_prob)   #输出为96*1024


# #密集连接层2
# W_fc2 = weight_variable([1024, 512])
# b_fc2 = bias_variable([512])

# h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop1, W_fc2) + b_fc2)

# #relu函数作用是求出max(h_fc1,0)
# h_fc1_drop2 = tf.nn.dropout(h_fc2, keep_prob)

# #输出层softmax层
# W_fc3 = weight_variable([512, 3])

# y_conv = tf.matmul(h_fc1_drop2, W_fc3)
# # tf.summary.histogram('y_conv',y_conv)

# # batch_x = images[1:3]
# # batch_y = labels[1:3]
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     print("shape of outputs is:",sess.run(tf.shape(y_conv),feed_dict={input_image:batch_x,
# #                                                                            y_:batch_y,keep_prob:1.0}))

# # cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# cross_entropy = tf.reduce_sum(tf.square(y_-y_conv))
# tf.summary.scalar("cross_entropy",cross_entropy)

# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# # tf.summary.scalar("accuracy", accuracy)
# merged = tf.summary.merge_all()

# #保存训练好的模型一遍下次使用
# saver = tf.train.Saver()


# batch_size = 2
# epochs = len(labels) // batch_size

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summary.FileWriter("logs_dir/", sess.graph)
#     for epoch in range(epochs):
#         for i in tqdm(range(len(images)-1)):
#             batch_x,batch_y = get_Batch(images, labels, batch_size)
#             summary,_ = sess.run([merged,train_step],feed_dict={input_image:batch_x, y_:batch_y, keep_prob:0.8})
#         writer.add_summary(summary, epoch)
#         #保存训练好的模型函数　第三个参数是想将训练的次数作为后缀加入到文件的名称中去
#         saver.save(sess, "net/save_net.ckpt",global_step=epoch)

