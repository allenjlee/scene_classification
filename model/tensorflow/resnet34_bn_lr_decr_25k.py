import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
from DataLoader_test import *

# Dataset Parameters
batch_size = 64
load_size = 128
fine_size = 112 # image size, 112x112x3
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.0001
# dropout = 0.55 # Dropout, probability to keep units
training_iters = 25000
step_display = 50
step_save = 5000
path_save = 'resnet_34_lr_bn_25k/'
start_from = ''

if not os.path.exists(path_save):
    os.makedirs(path_save)

def batch_norm_layer(x, train_phase, scope_bn):
    return batch_norm(x, decay=0.9, center=True, scale=True,
    updates_collections=None,
    is_training=train_phase,
    reuse=None,
    trainable=True,
    scope=scope_bn)


def resnet_34(x, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([7, 7, 3, 64], stddev=np.sqrt(2. / (7 * 7 * 3)))),  # 1
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2. / (3 * 3 * 64)))),  # 2
        'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2. / (3 * 3 * 64)))),  # 2
        'wc4': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2. / (3 * 3 * 64)))),  # 2
        'wc5': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2. / (3 * 3 * 64)))),  # 2
        'wc6': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2. / (3 * 3 * 64)))),  # 2
        'wc7': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2. / (3 * 3 * 64)))),  # 2

        'wc7a': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=np.sqrt(2. / (3 * 3 * 64)))), # 2a

        'wc8': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))),  # 3
        'wc9': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))),  # 3
        'wc10': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))),  # 3
        'wc11': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))),  # 3
        'wc12': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))),  # 3
        'wc13': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))),  # 3
        'wc14': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))),  # 3
        'wc15': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))),  # 3

        'wc15a': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=np.sqrt(2. / (3 * 3 * 128)))), # 3a

        'wc16': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc17': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc18': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc19': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc20': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc21': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc22': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc23': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc24': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc25': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc26': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4
        'wc27': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))),  # 4

        'wc27a': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=np.sqrt(2. / (3 * 3 * 256)))), # 4a

        'wc28': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=np.sqrt(2. / (3 * 3 * 512)))),  # 5
        'wc29': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2. / (3 * 3 * 512)))),  # 5
        'wc30': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2. / (3 * 3 * 512)))),  # 5
        'wc31': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2. / (3 * 3 * 512)))),  # 5
        'wc32': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2. / (3 * 3 * 512)))),  # 5
        'wc33': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2. / (3 * 3 * 512)))),  # 5


        'wo': tf.Variable(tf.random_normal([7*7*512, 100], stddev=np.sqrt(2./(7*7*512)))),
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # window of 112x112

    # output: 112x112     1 7x7 conv, 64 (remove stride 2 from first layer because input is half the size of 224)

    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = batch_norm_layer(conv1, train_phase, 'bn1')
    conv1 = tf.nn.relu(conv1)

    # output: 56x56       1 max pooling, /2
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # output: 56x56       4 3x3 conv, 64
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv2d(conv2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)
    conv3 += pool1

    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    conv5 = tf.nn.conv2d(conv4, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)
    conv5 += conv3

    conv6 = tf.nn.conv2d(conv5, weights['wc6'], strides=[1, 1, 1, 1], padding='SAME')
    conv6 = batch_norm_layer(conv6, train_phase, 'bn6')
    conv6 = tf.nn.relu(conv6)

    conv7 = tf.nn.conv2d(conv6, weights['wc7'], strides=[1, 1, 1, 1], padding='SAME')
    conv7 = batch_norm_layer(conv7, train_phase, 'bn7')
    conv7 = tf.nn.relu(conv7)
    conv7 += conv5

    # conv7 size fix for next residual add
    conv7_branch = tf.nn.conv2d(conv7, weights['wc7a'], strides=[1, 2, 2, 1], padding='SAME')
    conv7_branch = batch_norm_layer(conv7_branch, train_phase, 'bn7a')
    conv7_branch = tf.nn.relu(conv7_branch)


    # output: 28x28       4 3x3 conv, 128, /2 on first
    conv8 = tf.nn.conv2d(conv7, weights['wc8'], strides=[1, 2, 2, 1], padding='SAME')
    conv8 = batch_norm_layer(conv8, train_phase, 'bn8')
    conv8 = tf.nn.relu(conv8)

    conv9 = tf.nn.conv2d(conv8, weights['wc9'], strides=[1, 1, 1, 1], padding='SAME')
    conv9 = batch_norm_layer(conv9, train_phase, 'bn9')
    conv9 = tf.nn.relu(conv9)
    conv9 += conv7_branch #size mismatch

    conv10 = tf.nn.conv2d(conv9, weights['wc10'], strides=[1, 1, 1, 1], padding='SAME')
    conv10 = batch_norm_layer(conv10, train_phase, 'bn10')
    conv10 = tf.nn.relu(conv10)

    conv11 = tf.nn.conv2d(conv10, weights['wc11'], strides=[1, 1, 1, 1], padding='SAME')
    conv11 = batch_norm_layer(conv11, train_phase, 'bn11')
    conv11 = tf.nn.relu(conv11)
    conv11 += conv9

    conv12 = tf.nn.conv2d(conv11, weights['wc12'], strides=[1, 1, 1, 1], padding='SAME')
    conv12 = batch_norm_layer(conv12, train_phase, 'bn12')
    conv12 = tf.nn.relu(conv12)

    conv13 = tf.nn.conv2d(conv12, weights['wc13'], strides=[1, 1, 1, 1], padding='SAME')
    conv13 = batch_norm_layer(conv13, train_phase, 'bn13')
    conv13 = tf.nn.relu(conv13)
    conv13 += conv11

    conv14 = tf.nn.conv2d(conv13, weights['wc14'], strides=[1, 1, 1, 1], padding='SAME')
    conv14 = batch_norm_layer(conv14, train_phase, 'bn14')
    conv14 = tf.nn.relu(conv14)

    conv15 = tf.nn.conv2d(conv14, weights['wc15'], strides=[1, 1, 1, 1], padding='SAME')
    conv15 = batch_norm_layer(conv15, train_phase, 'bn15')
    conv15 = tf.nn.relu(conv15)
    conv15 += conv13

    # conv15 size fix for next residual add
    conv15_branch = tf.nn.conv2d(conv15, weights['wc15a'], strides=[1, 2, 2, 1], padding='SAME')
    conv15_branch = batch_norm_layer(conv15_branch, train_phase, 'bn15a')
    conv15_branch = tf.nn.relu(conv15_branch)

    # output: 14x14       4 3x3 conv, 256, /2 on first
    conv16 = tf.nn.conv2d(conv15, weights['wc16'], strides=[1, 2, 2, 1], padding='SAME')
    conv16 = batch_norm_layer(conv16, train_phase, 'bn16')
    conv16 = tf.nn.relu(conv16)

    conv17 = tf.nn.conv2d(conv16, weights['wc17'], strides=[1, 1, 1, 1], padding='SAME')
    conv17 = batch_norm_layer(conv17, train_phase, 'bn17')
    conv17 = tf.nn.relu(conv17)
    conv17 += conv15_branch #size mismatch

    conv18 = tf.nn.conv2d(conv17, weights['wc18'], strides=[1, 1, 1, 1], padding='SAME')
    conv18 = batch_norm_layer(conv18, train_phase, 'bn18')
    conv18 = tf.nn.relu(conv18)

    conv19 = tf.nn.conv2d(conv18, weights['wc19'], strides=[1, 1, 1, 1], padding='SAME')
    conv19 = batch_norm_layer(conv19, train_phase, 'bn19')
    conv19 = tf.nn.relu(conv19)
    conv19 += conv17

    conv20 = tf.nn.conv2d(conv19, weights['wc20'], strides=[1, 1, 1, 1], padding='SAME')
    conv20 = batch_norm_layer(conv20, train_phase, 'bn20')
    conv20 = tf.nn.relu(conv20)

    conv21 = tf.nn.conv2d(conv20, weights['wc21'], strides=[1, 1, 1, 1], padding='SAME')
    conv21 = batch_norm_layer(conv21, train_phase, 'bn21')
    conv21 = tf.nn.relu(conv21)
    conv21 += conv19

    conv22 = tf.nn.conv2d(conv21, weights['wc22'], strides=[1, 1, 1, 1], padding='SAME')
    conv22 = batch_norm_layer(conv22, train_phase, 'bn22')
    conv22 = tf.nn.relu(conv22)

    conv23 = tf.nn.conv2d(conv22, weights['wc23'], strides=[1, 1, 1, 1], padding='SAME')
    conv23 = batch_norm_layer(conv23, train_phase, 'bn23')
    conv23 = tf.nn.relu(conv23)
    conv23 += conv21

    conv24 = tf.nn.conv2d(conv23, weights['wc24'], strides=[1, 1, 1, 1], padding='SAME')
    conv24 = batch_norm_layer(conv24, train_phase, 'bn24')
    conv24 = tf.nn.relu(conv24)

    conv25 = tf.nn.conv2d(conv24, weights['wc25'], strides=[1, 1, 1, 1], padding='SAME')
    conv25 = batch_norm_layer(conv25, train_phase, 'bn25')
    conv25 = tf.nn.relu(conv25)
    conv25 += conv23

    conv26 = tf.nn.conv2d(conv25, weights['wc26'], strides=[1, 1, 1, 1], padding='SAME')
    conv26 = batch_norm_layer(conv26, train_phase, 'bn26')
    conv26 = tf.nn.relu(conv26)

    conv27 = tf.nn.conv2d(conv26, weights['wc27'], strides=[1, 1, 1, 1], padding='SAME')
    conv27 = batch_norm_layer(conv27, train_phase, 'bn27')
    conv27 = tf.nn.relu(conv27)
    conv27 += conv25

    # conv13 size fix for next residual add
    conv27_branch = tf.nn.conv2d(conv27, weights['wc27a'], strides=[1, 2, 2, 1], padding='SAME')
    conv27_branch = batch_norm_layer(conv27_branch, train_phase, 'bn27a')
    conv27_branch = tf.nn.relu(conv27_branch)

    # output: 7x7         4 3x3 conv, 512, /2 on first
    conv28 = tf.nn.conv2d(conv27, weights['wc28'], strides=[1, 2, 2, 1], padding='SAME')
    conv28 = batch_norm_layer(conv28, train_phase, 'bn28')
    conv28 = tf.nn.relu(conv28)

    conv29 = tf.nn.conv2d(conv28, weights['wc29'], strides=[1, 1, 1, 1], padding='SAME')
    conv29 = batch_norm_layer(conv29, train_phase, 'bn29')
    conv29 = tf.nn.relu(conv29)
    conv29 += conv27_branch #size mismatch

    conv30 = tf.nn.conv2d(conv29, weights['wc30'], strides=[1, 1, 1, 1], padding='SAME')
    conv30 = batch_norm_layer(conv30, train_phase, 'bn30')
    conv30 = tf.nn.relu(conv30)

    conv31 = tf.nn.conv2d(conv30, weights['wc31'], strides=[1, 1, 1, 1], padding='SAME')
    conv31 = batch_norm_layer(conv31, train_phase, 'bn31')
    conv31 = tf.nn.relu(conv31)
    conv31 += conv29

    conv32 = tf.nn.conv2d(conv31, weights['wc32'], strides=[1, 1, 1, 1], padding='SAME')
    conv32 = batch_norm_layer(conv32, train_phase, 'bn32')
    conv32 = tf.nn.relu(conv32)

    conv33 = tf.nn.conv2d(conv32, weights['wc33'], strides=[1, 1, 1, 1], padding='SAME')
    conv33 = batch_norm_layer(conv33, train_phase, 'bn33')
    conv33 = tf.nn.relu(conv33)
    conv33 += conv31

    # output: 7x7         1 avg pooling
    pool2 = tf.nn.avg_pool(conv33, ksize=[1, 3, 3, 1], strides=[1,1,1,1], padding='SAME')

    # output: 100         1 fully-connected layer output 100 (7*7*512, 100)
    out = tf.reshape(pool2, [-1, weights['wo'].get_shape().as_list()[0]])
    out = tf.add(tf.matmul(out, weights['wo']), biases['bo'])


    return out

def lrn(x, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)

# Construct dataloader
opt_data_train = {
    #'data_h5': 'miniplaces_256_train.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/train.txt', # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': True
    }

opt_data_val = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/val.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
    }

opt_data_test = {
    #'data_h5': 'miniplaces_256_val.h5',
    'data_root': '../../data/images/',   # MODIFY PATH ACCORDINGLY
    'data_list': '../../data/test.txt',   # MODIFY PATH ACCORDINGLY
    'load_size': load_size,
    'fine_size': fine_size,
    'data_mean': data_mean,
    'randomize': False
}

loader_train = DataLoaderDisk(**opt_data_train)
loader_val = DataLoaderDisk(**opt_data_val)
loader_test = DataLoaderDisk_t(**opt_data_test)

# tf Graph input
x = tf.placeholder(tf.float32, [None, fine_size, fine_size, c])
y = tf.placeholder(tf.int64, None)
train_phase = tf.placeholder(tf.bool)

# Construct model
logits = resnet_34(x, train_phase)

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluate model
accuracy1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 1), tf.float32))
accuracy5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, y, 5), tf.float32))
top5 = tf.nn.top_k(logits, 5)


# define initialization
init = tf.global_variables_initializer()

# define saver
saver = tf.train.Saver()

# define summary writer
#writer = tf.train.SummaryWriter('.', graph=tf.get_default_graph())

# Launch the graph
with tf.Session() as sess:
    # Initialization
    if len(start_from)>1:
        saver = tf.train.import_meta_graph('')
        saver.restore(sess, tf.train.latest_checkpoint("resnet_18_3_bn/"))
    else:
        sess.run(init)
    print('training_iters', training_iters)
    print('learning_rate', learning_rate)

    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        
        if step == 20000:
            sess.run(train_optimizer, feed_dict={learning_rate: 0.00001})
            print("changed learning_rate")

        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, train_phase: False})
            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, train_phase: False})
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))
			
        # if acc5 > 0.80:
        #     break
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, train_phase: True})
        
        step += 1
        
        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))
        
    print("Optimization Finished!")


    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, train_phase: False})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))
	
    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))

    print('Evaluating on test set...')
    outpt = open('allenlee.resnet34_bn_lr_decr.pred.txt', 'w')
    test_num_batch = loader_test.size()
    loader_test.reset()

    for j in range(test_num_batch):
        test_img_batch = loader_test.next_batch(1)
        test_img_lab = "test/" + "%08d" % (j+1,) + ".jpg"
        #print(test_img_batch)i
        # for img in test_img_batch:
        #     #calculate logits (aka final layer output)
        #     #keep running sum of all logits

        res = sess.run([top5], feed_dict={x: test_img_batch, train_phase: False})[0][1][0]

        #Compute top 5 labels basedon this summed logits (should reflect the average of all predictions from 10 crops)

        for r in res:
            test_img_lab = test_img_lab + " " + str(r)
        print (test_img_lab)
        outpt.write(test_img_lab + "\n")
    outpt.close()



