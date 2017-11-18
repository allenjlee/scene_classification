import os, datetime
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from DataLoader import *
from DataLoader_test import *

# Dataset Parameters
batch_size = 32
load_size = 128 # image size is 128
fine_size = 112 # image size, 224 x 224 x 3
c = 3
data_mean = np.asarray([0.45834960097,0.44674252445,0.41352266842])

# Training Parameters
learning_rate = 0.00001
dropout = 0.55 # Dropout, probability to keep units
training_iters = 10000
step_display = 50
step_save = 5000
path_save = 'vgg_19_results_test/'
start_from = 'True'

if not os.path.exists(path_save):
    os.makedirs(path_save)



def vgg_19(x, keep_dropout):
    weights = {
        'wc1': tf.Variable(tf.random_normal([3, 3, 3, 64], stddev=np.sqrt(2./(3*3*3)))),
        'wc2': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2./(3*3*64)))),
        'wc3': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=np.sqrt(2./(3*3*64)))),
        'wc4': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2./(3*3*128)))),
        'wc5': tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=np.sqrt(2./(3*3*128)))),
        'wc6': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2./(3*3*256)))),
        'wc7': tf.Variable(tf.random_normal([3, 3, 256, 512], stddev=np.sqrt(2./(3*3*256)))),
        'wc8': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*256)))),
        'wc9': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc10': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc11': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc12': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc13': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),
        'wc14': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2./(3*3*512)))),


        'wf15': tf.Variable(tf.random_normal([7*7*512, 4096], stddev=np.sqrt(2./(7*7*512)))),
        'wf16': tf.Variable(tf.random_normal([4096, 4096], stddev=np.sqrt(2./4096))),
        'wo': tf.Variable(tf.random_normal([4096, 100], stddev=np.sqrt(2./4096)))
    }

    biases = {
        'bo': tf.Variable(tf.ones(100))
    }

    # first pool

    conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.nn.conv2d(conv1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    pool1 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # second pool

    conv3 = tf.nn.conv2d(pool1, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = tf.nn.relu(conv3)

    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    pool2 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # third pool

    conv5 = tf.nn.conv2d(pool2, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = tf.nn.relu(conv5)

    conv6 = tf.nn.conv2d(conv5, weights['wc6'], strides=[1, 1, 1, 1], padding='SAME')
    conv6 = tf.nn.relu(conv6)
    
    pool3 = tf.nn.max_pool(conv6, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
    

    # fourth pool

    conv7 = tf.nn.conv2d(pool3, weights['wc7'], strides=[1, 1, 1, 1], padding='SAME')
    conv7 = tf.nn.relu(conv7)

    conv8 = tf.nn.conv2d(conv7, weights['wc8'], strides=[1, 1, 1, 1], padding='SAME')
    conv8 = tf.nn.relu(conv8)

    conv9 = tf.nn.conv2d(conv8, weights['wc9'], strides=[1, 1, 1, 1], padding='SAME')
    conv9 = tf.nn.relu(conv9)
    

    conv10 = tf.nn.conv2d(conv9, weights['wc10'], strides=[1, 1, 1, 1], padding='SAME')
    conv10 = tf.nn.relu(conv10)
    pool4 = tf.nn.max_pool(conv10, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # fifth pool

    conv11 = tf.nn.conv2d(pool4, weights['wc11'], strides=[1, 1, 1, 1], padding='SAME')
    conv11 = tf.nn.relu(conv11)

    conv12 = tf.nn.conv2d(conv11, weights['wc12'], strides=[1, 1, 1, 1], padding='SAME')
    conv12 = tf.nn.relu(conv12)

    conv13 = tf.nn.conv2d(conv12, weights['wc13'], strides=[1, 1, 1, 1], padding='SAME')
    conv13 = tf.nn.relu(conv13)

    conv14 = tf.nn.conv2d(conv13, weights['wc14'], strides=[1, 1, 1, 1], padding='SAME')
    conv14 = tf.nn.relu(conv14)

    pool5 = tf.nn.max_pool(conv14, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

    # first FC

    fc14 = tf.reshape(pool5, [-1, weights['wf15'].get_shape().as_list()[0]])
    fc14 = tf.matmul(fc14, weights['wf15'])
    fc14 = tf.nn.relu(fc14)
    fc14 = tf.nn.dropout(fc14, keep_dropout)

    # FC + ReLU + Dropout
    fc15 = tf.matmul(fc14, weights['wf16'])
    fc15 = tf.nn.relu(fc15)
    fc15 = tf.nn.dropout(fc15, keep_dropout)

    # Output FC
    out = tf.add(tf.matmul(fc15, weights['wo']), biases['bo'])

    return out

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
keep_dropout = tf.placeholder(tf.float32)


# Construct model
logits = vgg_19(x, keep_dropout)

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
        saver = tf.train.import_meta_graph('-40000.meta')
        saver.restore(sess, tf.train.latest_checkpoint("vgg_19_results_1/"))
    else:
        sess.run(init)
    
    step = 10000000

    while step < training_iters:

        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        
        if step % step_display == 0:
            print('[%s]:' %(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

            # Calculate batch loss and accuracy on training set
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.}) 
            print("-Iter " + str(step) + ", Training Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))

            # Calculate batch loss and accuracy on validation set
            images_batch_val, labels_batch_val = loader_val.next_batch(batch_size)    
            l, acc1, acc5 = sess.run([loss, accuracy1, accuracy5], feed_dict={x: images_batch_val, y: labels_batch_val, keep_dropout: 1.}) 
            print("-Iter " + str(step) + ", Validation Loss= " + \
                  "{:.6f}".format(l) + ", Accuracy Top1 = " + \
                  "{:.4f}".format(acc1) + ", Top5 = " + \
                  "{:.4f}".format(acc5))
			
        if acc5 > 0.80:
            break
        # Run optimization op (backprop)
        sess.run(train_optimizer, feed_dict={x: images_batch, y: labels_batch, keep_dropout: dropout})
        
        step += 1
        
        # Save model
        if step % step_save == 0:
            saver.save(sess, path_save, global_step=step)
            print("Model saved at Iter %d !" %(step))
        
    print("Optimization Finished!")


    # Evaluate on the whole validation set
    print('Evaluation on the whole validation set...')
    num_batch = loader_val.size()//batch_size
    print(loader_val.size)
    acc1_total = 0.
    acc5_total = 0.
    loader_val.reset()
    for i in range(num_batch):
        images_batch, labels_batch = loader_val.next_batch(batch_size)    
        acc1, acc5 = sess.run([accuracy1, accuracy5], feed_dict={x: images_batch, y: labels_batch, keep_dropout: 1.})
        acc1_total += acc1
        acc5_total += acc5
        print("Validation Accuracy Top1 = " + \
              "{:.4f}".format(acc1) + ", Top5 = " + \
              "{:.4f}".format(acc5))
	
    acc1_total /= num_batch
    acc5_total /= num_batch
    print('Evaluation Finished! Accuracy Top1 = ' + "{:.4f}".format(acc1_total) + ", Top5 = " + "{:.4f}".format(acc5_total))

    print('Evaluating on test set...')
    outpt = open('allenlee.test.pred.txt', 'w')
    test_num_batch = loader_test.size()
    loader_test.reset()

    for j in range(test_num_batch):
        test_img_batch = loader_test.next_batch(1)
        test_img_lab = "test/" + "%08d" % (j+1,) + ".jpg"
        #print(test_img_batch)
        res = sess.run([top5], feed_dict = {x: test_img_batch, keep_dropout: 1.})[0][1][0]
        for r in res:
            test_img_lab = test_img_lab + " " + str(r)
        print (test_img_lab)
        outpt.write(test_img_lab + "\n")
    outpt.close()



