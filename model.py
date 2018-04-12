#coding=utf-8
import tensorflow as tf

#卷积层
def conv2(input_data, w_name, b_name, w_shape, b_shape, stride, initializer_weitht, initializer_biases):

    weights = tf.get_variable(name=w_name, shape=w_shape, initializer=initializer_weitht)
    biases = tf.get_variable(name=b_name, shape=b_shape, initializer=initializer_biases)
    conv = tf.nn.conv2d(input=input_data, filter=weights, strides=stride, padding='SAME')
    out_data = tf.nn.relu(tf.nn.bias_add(conv, biases))
    return out_data

#池化层
def pool(input_data, name, ksize, strides):

    return tf.nn.max_pool(input_data, ksize=ksize, strides=strides, padding='SAME', name=name)


#全连接层，在此直接设置dropout的系数,从卷积层的input_data注意reshape
def full_connec(input_data, w_name, b_name, w_shape, b_shape, keep_prob, initializer_weitht, initializer_biases):

    weights = tf.get_variable(name=w_name, shape=w_shape, initializer=initializer_weitht)
    biases = tf.get_variable(name=b_name, shape=b_shape, initializer=initializer_biases)
    full_data = tf.nn.relu(tf.matmul(input_data, weights) + biases)
    out_data = tf.nn.dropout(x=full_data, keep_prob=keep_prob)
    return out_data

def custom_softmax(input_data, w_name, b_name, w_shape, b_shape, initializer_weitht, initializer_biases):
    weights = tf.get_variable(name=w_name, shape=w_shape, initializer=initializer_weitht)
    biases = tf.get_variable(name=b_name, shape=b_shape, initializer=initializer_biases)
    logit = tf.matmul(input_data, weights) + biases
    class_score = tf.nn.softmax(logits=logit)
    return class_score

def inference(S_images, T_images, batch_size, n_classes):

    # 空间网络
    s_conv_1 = conv2(input_data=S_images,
                   w_name='s_weight_1',
                   b_name='s_biase_1',
                   w_shape=[7, 7, 3, 96],
                   b_shape=[96],
                   stride=[1, 2, 2, 1],
                   initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                   initializer_biases=tf.constant_initializer(0.1))
    s_outdata_conv1 = pool(tf.nn.relu(s_conv_1), name='s_conv1_pool', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    s_conv_2 = conv2(input_data=s_outdata_conv1,
                   w_name='s_weight_2',
                   b_name='s_biase_2',
                   w_shape=[5, 5, 96, 256],
                   b_shape=[256],
                   stride=[1, 2, 2, 1],
                   initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                   initializer_biases=tf.constant_initializer(0.1))
    s_outdata_conv2 = pool(tf.nn.relu(s_conv_2), name='s_conv2_pool', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    s_conv_3 = conv2(input_data=s_outdata_conv2,
                   w_name='s_weight_3',
                   b_name='s_biase_3',
                   w_shape=[5, 5, 256, 512],
                   b_shape=[512],
                   stride=[1, 1, 1, 1],
                   initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                   initializer_biases=tf.constant_initializer(0.1))

    s_conv_4 = conv2(input_data=s_conv_3,
                   w_name='s_weight_4',
                   b_name='s_biase_4',
                   w_shape=[5, 5, 512, 512],
                   b_shape=[512],
                   stride=[1, 1, 1, 1],
                   initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                   initializer_biases=tf.constant_initializer(0.1))

    s_conv_5 = conv2(input_data=s_conv_4,
                   w_name='s_weight_5',
                   b_name='s_biase_5',
                   w_shape=[5, 5, 512, 512],
                   b_shape=[512],
                   stride=[1, 1, 1, 1],
                   initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                   initializer_biases=tf.constant_initializer(0.1))
    s_outdata_conv5 = pool(tf.nn.relu(s_conv_5), name='s_conv5_pool', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    # 空间图像全连接层
    s_reshape = tf.reshape(s_outdata_conv5, shape=[batch_size, -1])
    dim = s_reshape.get_shape()[1].value
    s_outdata_fc1 = full_connec(input_data=s_reshape,
                w_name='s_full_weight_1',
                b_name='s_full_biase_1',
                w_shape=[dim, 4096],
                b_shape=[4096],
                keep_prob=0.8,
                initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                initializer_biases=tf.constant_initializer(0.1))

    s_outdata_fc2 = full_connec(input_data=s_outdata_fc1,
                w_name='s_full_weight_2',
                b_name='s_full_biase_2',
                w_shape=[4096, 2048],
                b_shape=[2048],
                keep_prob=0.8,
                initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                initializer_biases=tf.constant_initializer(0.1))

    # softmax层后空间网络的分类结果
    s_class_score = custom_softmax(input_data=s_outdata_fc2,
                                   w_name='s_softmax_weight',
                                   b_name='s_softmax_biase',
                                   w_shape=[2048, n_classes],
                                   b_shape=[n_classes],
                                   initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                   initializer_biases=tf.constant_initializer(0.1)
                                   )



    # 时序网络
    t_conv_1 = conv2(input_data=T_images,
                     w_name='t_weight_1',
                     b_name='t_biase_1',
                     w_shape=[7, 7, 20, 96],
                     b_shape=[96],
                     stride=[1, 2, 2, 1],
                     initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                     initializer_biases=tf.constant_initializer(0.1))
    t_outdata_conv1 = pool(tf.nn.relu(t_conv_1), name='t_conv1_pool', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    t_conv_2 = conv2(input_data=t_outdata_conv1,
                     w_name='t_weight_2',
                     b_name='t_biase_2',
                     w_shape=[5, 5, 96, 256],
                     b_shape=[256],
                     stride=[1, 2, 2, 1],
                     initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                     initializer_biases=tf.constant_initializer(0.1))
    t_outdata_conv2 = pool(tf.nn.relu(t_conv_2), name='t_conv2_pool', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    t_conv_3 = conv2(input_data=t_outdata_conv2,
                     w_name='t_weight_3',
                     b_name='t_biase_3',
                     w_shape=[5, 5, 256, 512],
                     b_shape=[512],
                     stride=[1, 1, 1, 1],
                     initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                     initializer_biases=tf.constant_initializer(0.1))

    t_conv_4 = conv2(input_data=t_conv_3,
                     w_name='t_weight_4',
                     b_name='t_biase_4',
                     w_shape=[5, 5, 512, 512],
                     b_shape=[512],
                     stride=[1, 1, 1, 1],
                     initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                     initializer_biases=tf.constant_initializer(0.1))

    t_conv_5 = conv2(input_data=t_conv_4,
                     w_name='t_weight_5',
                     b_name='t_biase_5',
                     w_shape=[5, 5, 512, 512],
                     b_shape=[512],
                     stride=[1, 1, 1, 1],
                     initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                     initializer_biases=tf.constant_initializer(0.1))
    t_outdata_conv5 = pool(tf.nn.relu(t_conv_5), name='t_conv5_pool', ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1])

    # 空间图像全连接层
    t_reshape = tf.reshape(t_outdata_conv5, shape=[batch_size, -1])
    dim = t_reshape.get_shape()[1].value
    t_outdata_fc1 = full_connec(input_data=t_reshape,
                                w_name='t_full_weight_1',
                                b_name='t_full_biase_1',
                                w_shape=[dim, 4096],
                                b_shape=[4096],
                                keep_prob=0.8,
                                initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                initializer_biases=tf.constant_initializer(0.1))

    t_outdata_fc2 = full_connec(input_data=t_outdata_fc1,
                                w_name='t_full_weight_2',
                                b_name='t_full_biase_2',
                                w_shape=[4096, 2048],
                                b_shape=[2048],
                                keep_prob=0.8,
                                initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                initializer_biases=tf.constant_initializer(0.1))

    # softmax层后空间网络的分类结果
    t_class_score = custom_softmax(input_data=t_outdata_fc2,
                                   w_name='t_softmax_weight',
                                   b_name='t_softmax_biase',
                                   w_shape=[2048, n_classes],
                                   b_shape=[n_classes],
                                   initializer_weitht=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32),
                                   initializer_biases=tf.constant_initializer(0.1)
                                   )
    total_class_score = s_class_score + t_class_score

    return total_class_score


def losses(logits, labels):

    cross_entropy = -tf.reduce_mean(labels*tf.log(logits), name='loss')

    return cross_entropy

#
def trainning(loss, learning_rate):

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

# 之后需要根据label的存储方式(是稀疏还是非稀疏)，来适当修改计算方式。
def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    correct = tf.cast(correct, tf.float16)
    accuracy = tf.reduce_mean(correct)
    return accuracy
