import os
import random
import skimage.data
from skimage import io
import skimage.transform as skt
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

train_label_dir = "/home/ysk/database/train_data/benhuan/"
train_pic_dir = "/home/ysk/database/train_data/benhuan//picture"
train_submap_dir = "/home/ysk/database/train_data/benhuan//submap"


def get_train_data_batch(num, pic_dir=train_pic_dir, label_dir=train_label_dir, submap_dir=train_submap_dir):
    train_txt = os.path.join(label_dir, "label.txt")
    count = len(open(train_txt, 'r').readlines()) - 2
    idx = np.arange(0, count)
    np.random.shuffle(idx)
    data_idx = idx[:num]
    #     print(data_idx)

    labels = []
    images = []

    for i in range(num):
        with open(train_txt, 'r') as file:
            s = file.readlines()
            image1 = cv2.imread(os.path.join(pic_dir, s[data_idx[i]].split(";")[0]))
            submap1 = cv2.imread(os.path.join(submap_dir, str(data_idx[i]) + '.pgm'))
            submap1 = cv2.resize(submap1, (448, 448))
            submap1 = cv2.transpose(submap1)
            submap1 = cv2.flip(submap1, 1)

            #             image2 = skimage.data.imread(os.path.join(pic_dir, s[data_idx[i] + 1].split(";")[0]))
            #             submap2 = cv2.imread(os.path.join(submap_dir,str(data_idx[i] + 1) + '.pgm'))
            #             submap2 = cv2.resize(submap2,(448,448))
            #             submap2 = cv2.transpose(submap2)
            #             submap2 = cv2.flip(submap2, 1)

            img = cv2.merge([image1, submap1])
            images.append(img)

            label = s[data_idx[i]].split(";")[1:]
            label = list(map(float, label))
            labels.append(label)
    return np.asarray(images), np.asarray(labels)


with tf.name_scope("input"):
    input_image = tf.placeholder("float", [None, 448, 448, 6], name='input_image') / 255. * 2. - 1
    y_ = tf.placeholder("float", [None, 3], name='y_')


def weight_variable(shape):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(0.5)(w))
    return w


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 第0层卷积
with tf.name_scope("conv_0"):
    W_conv0 = weight_variable([7, 7, 6, 64])
    b_conv0 = bias_variable([64])
    conv_layer0 = tf.nn.conv2d(input_image, W_conv0, strides=[1, 1, 1, 1], padding="SAME") + b_conv0
    conv_layer_nor0 = tf.layers.batch_normalization(conv_layer0, training=True)

    h_conv0 = tf.nn.leaky_relu(conv_layer_nor0)
with tf.name_scope("max_pool_0"):
    h_pool0 = max_pool_2x2(h_conv0)  # 输出变成224*224*64 [None,224,224,64]

# #第一层卷积层
# #参数的前两维为卷积核的维度，第三个参数为当前层的深度，第四个为输出到层的深度
with tf.name_scope("conv_1"):
    W_conv1 = weight_variable([5, 5, 64, 128])
    b_conv1 = bias_variable([128])
    conv_layer1 = conv2d(h_pool0, W_conv1) + b_conv1
    conv_layer_nor1 = tf.layers.batch_normalization(conv_layer1, training=True)

    h_conv1 = tf.nn.leaky_relu(conv_layer_nor1)
with tf.name_scope("max_pool1"):
    h_pool1 = max_pool_2x2(h_conv1)  # 输出变成112*112*64   [None,112,112,64]

# 第二层卷积层
with tf.name_scope("conv_2"):
    W_conv2 = weight_variable([5, 5, 128, 192])
    b_conv2 = bias_variable([192])
    conv_layer2 = conv2d(h_pool1, W_conv2) + b_conv2
    conv_layer_nor2 = tf.layers.batch_normalization(conv_layer2, training=True)

    h_conv2 = tf.nn.leaky_relu(conv_layer_nor2)
with tf.name_scope("max_pool_2"):
    h_pool2 = max_pool_2x2(h_conv2)  # 输出变为56*56*192

# 第三层卷积层
with tf.name_scope("conv_3"):
    W_conv3 = weight_variable([3, 3, 192, 128])
    b_conv3 = bias_variable([128])
    conv_layer3 = conv2d(h_pool2, W_conv3) + b_conv3
    conv_layer_nor3 = tf.layers.batch_normalization(conv_layer3, training=True)

    h_conv3 = tf.nn.leaky_relu(conv_layer_nor3)
with tf.name_scope("max_pool_3"):
    h_pool3 = max_pool_2x2(h_conv3)  # 输出为28*28*128

# 第四层卷积层
with tf.name_scope("conv_4"):
    W_conv4 = weight_variable([3, 3, 128, 64])
    b_conv4 = bias_variable([64])
    conv_layer4 = conv2d(h_pool3, W_conv4) + b_conv4
    conv_layer_nor4 = tf.layers.batch_normalization(conv_layer4, training=True)

    h_conv4 = tf.nn.leaky_relu(conv_layer_nor4)
with tf.name_scope("max_pool_4"):
    h_pool4 = max_pool_2x2(h_conv4)  # 输出变为14*14*64

keep_prob = tf.placeholder("float")

# 密集连接层0
with tf.name_scope("fc0"):
    W_fc0 = weight_variable([14 * 14 * 64, 2048])
    b_fc0 = bias_variable([2048])

    h_pool4_flat = tf.reshape(h_pool4, [-1, 14 * 14 * 64])

    out_fc0 = tf.matmul(h_pool4_flat, W_fc0) + b_fc0
    out_nor_fc0 = tf.layers.batch_normalization(out_fc0, training=True)

    h_fc0 = tf.nn.leaky_relu(out_nor_fc0)  # 输出为96*2048

    # relu函数作用是求出max(h_fc1,0)
    h_fc1_drop0 = tf.layers.dropout(h_fc0, keep_prob)  # 输出为96*2048

# 密集连接层1
with tf.name_scope("fc1"):
    W_fc1 = weight_variable([2048, 1024])
    b_fc1 = bias_variable([1024])

    out_fc1 = tf.matmul(h_fc1_drop0, W_fc1) + b_fc1
    out_nor_fc1 = tf.layers.batch_normalization(out_fc1, training=True)

    h_fc1 = tf.nn.leaky_relu(out_nor_fc1)  # 输出为2048*1024

    # relu函数作用是求出max(h_fc1,0)
    h_fc1_drop1 = tf.layers.dropout(h_fc1, keep_prob)  # 输出为2048*1024

# 密集连接层2
with tf.name_scope("fc2"):
    W_fc2 = weight_variable([1024, 512])
    b_fc2 = bias_variable([512])

    out_fc2 = tf.matmul(h_fc1_drop1, W_fc2) + b_fc2
    out_nor_fc2 = tf.layers.batch_normalization(out_fc2, training=True)

    h_fc2 = tf.nn.leaky_relu(out_nor_fc2)
    # relu函数作用是求出max(h_fc1,0)
    h_fc1_drop2 = tf.layers.dropout(h_fc2, keep_prob)

# 输出层
with tf.name_scope("out_layer"):
    W_fc3 = weight_variable([512, 3])
    y_conv = tf.matmul(h_fc1_drop2, W_fc3)


if __name__ == '__main__':
    starter_learning_rate = 0.3
    steps_per_decay = 3000
    decay_factor = 0.96

    global_step = tf.Variable(tf.constant(0))
    learning_rate = tf.train.exponential_decay(learning_rate=starter_learning_rate,
                                               global_step=global_step,
                                               decay_steps=steps_per_decay,
                                               decay_rate=decay_factor,
                                               staircase=True)

    delta = tf.constant(1.25)
    # huber损失函数

    loss1 = tf.reduce_mean(tf.multiply(tf.square(delta),
                                       tf.sqrt(1. + tf.square((y_ - y_conv) / delta)) - 1.))
    # loss = loss1 + tf.add_n(tf.get_collection('losses'))
    loss = loss1

    tf.summary.scalar("loss", loss)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    merged = tf.summary.merge_all()

    batch_size = 6
    checkpointsPath = "/home/ysk/database/train_data/benhuan/fc_bn_checkpoints_4805/"
    reload = True

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        var_list = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        var_list += bn_moving_vars
        saver = tf.train.Saver(var_list=var_list, max_to_keep=5)

        writer = tf.summary.FileWriter("logs_dir/", sess.graph)

        if not os.path.exists(checkpointsPath):
            os.mkdir(checkpointsPath)

        if reload:
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            if checkPoint and checkPoint.model_checkpoint_path:

                #             saver = tf.train.Saver(var_list=var_list)
                #             saver.restore(sess, checkPoint.model_checkpoint_path)

                #             var = tf.global_variables()
                var_to_restore = [val for val in var_list if 'fc0' not in val.name or
                                  'fc1' not in val.name or 'fc2' not in val.name or 'out_layer' not in val.name]
                saver = tf.train.Saver(var_to_restore)
                saver.restore(sess, checkPoint.model_checkpoint_path)
                var_to_init = [val for val in var_list if 'fc0' in val.name or
                               'fc1' in val.name or 'fc2' in val.name or 'out_layer' in val.name]
                tf.variables_initializer(var_to_init)
                print("restored %s" % checkPoint.model_checkpoint_path)
            else:
                print("no checkpoint found!")
        else:
            sess.run(tf.global_variables_initializer())

        for i in tqdm(range(200000)):
            batch_x, batch_y = get_train_data_batch(batch_size)
            batch_y = batch_y * 10000
            summary, _ = sess.run([merged, train_step], feed_dict={global_step: i,
                                                                   input_image: batch_x,
                                                                   y_: batch_y,
                                                                   keep_prob: 0.7})
            writer.add_summary(summary, i)
            if (i % 1000) == 0 & (i != 0):
                print("save model")
                print("learning_rate is:\n", sess.run(learning_rate, feed_dict={global_step: i}))
                print("real location is:\n", batch_y)
                print("prediction is:\n", sess.run(y_conv, feed_dict={input_image: batch_x,
                                                                      y_: batch_y, keep_prob: 1.0}))

                saver.save(sess, os.path.join(checkpointsPath, "saved_net.ckpt"), global_step=i)
                print("loss is:", sess.run(loss, feed_dict={input_image: batch_x, y_: batch_y, keep_prob: 1.0}))
        print("done!")