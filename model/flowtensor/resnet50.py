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
training_iters = 20000
step_display = 50
step_save = 5000
path_save = 'resnet50_results_3/'
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


def resnet_50(x, train_phase):
    weights = {
        'wc1': tf.Variable(tf.random_normal([7, 7, 3, 64], stddev=np.sqrt(2. / (7 * 7 * 3)))),  # 1

        'wc2': tf.Variable(tf.random_normal([1, 1, 64, 64], stddev=np.sqrt(2. / (1 * 1 * 64)))),  # 2
        'wc3': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2. / (3 * 3 * 64)))),  # 2
        'wc4': tf.Variable(tf.random_normal([1, 1, 64, 256], stddev=np.sqrt(2. / (1 * 1 * 64)))), # 2
        'wc5': tf.Variable(tf.random_normal([1, 1, 256, 64], stddev=np.sqrt(2. / (1 * 1 * 64)))),  # 2
        'wc6': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2. / (3 * 3 * 64)))),  # 2
        'wc7': tf.Variable(tf.random_normal([1, 1, 64, 256], stddev=np.sqrt(2. / (1 * 1 * 64)))), # 2
        'wc8': tf.Variable(tf.random_normal([1, 1, 256, 64], stddev=np.sqrt(2. / (1 * 1 * 64)))),  # 2
        'wc9': tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=np.sqrt(2. / (3 * 3 * 64)))),  # 2
        'wc10': tf.Variable(tf.random_normal([1, 1, 64, 256], stddev=np.sqrt(2. / (1 * 1 * 64)))), # 2

        'wc2a': tf.Variable(tf.random_normal([1, 1, 64, 256], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 2a

        'wc11': tf.Variable(tf.random_normal([1, 1, 256, 128], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 3
        'wc12': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))), # 3
        'wc13': tf.Variable(tf.random_normal([1, 1, 128, 512], stddev=np.sqrt(2. / (1 * 1 * 128)))), # 3
        'wc14': tf.Variable(tf.random_normal([1, 1, 512, 128], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 3
        'wc15': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))), # 3
        'wc16': tf.Variable(tf.random_normal([1, 1, 128, 512], stddev=np.sqrt(2. / (1 * 1 * 128)))), # 3
        'wc17': tf.Variable(tf.random_normal([1, 1, 512, 128], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 3
        'wc18': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))), # 3
        'wc19': tf.Variable(tf.random_normal([1, 1, 128, 512], stddev=np.sqrt(2. / (1 * 1 * 128)))), # 3
        'wc20': tf.Variable(tf.random_normal([1, 1, 512, 128], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 3
        'wc21': tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=np.sqrt(2. / (3 * 3 * 128)))), # 3
        'wc22': tf.Variable(tf.random_normal([1, 1, 128, 512], stddev=np.sqrt(2. / (1 * 1 * 128)))), # 3

        'wc11a': tf.Variable(tf.random_normal([1, 1, 256, 512], stddev=np.sqrt(2. / (1 * 1 * 512)))), # 3a

        'wc23': tf.Variable(tf.random_normal([1, 1, 512, 256], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 4
        'wc24': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))), # 4
        'wc25': tf.Variable(tf.random_normal([1, 1, 256, 1024], stddev=np.sqrt(2. / (1 * 1 * 1024)))), # 4
        'wc26': tf.Variable(tf.random_normal([1, 1, 1024, 256], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 4
        'wc27': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))), # 4
        'wc28': tf.Variable(tf.random_normal([1, 1, 256, 1024], stddev=np.sqrt(2. / (1 * 1 * 1024)))), # 4
        'wc29': tf.Variable(tf.random_normal([1, 1, 1024, 256], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 4
        'wc30': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))), # 4
        'wc31': tf.Variable(tf.random_normal([1, 1, 256, 1024], stddev=np.sqrt(2. / (1 * 1 * 1024)))), # 4
        'wc32': tf.Variable(tf.random_normal([1, 1, 1024, 256], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 4
        'wc33': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))), # 4
        'wc34': tf.Variable(tf.random_normal([1, 1, 256, 1024], stddev=np.sqrt(2. / (1 * 1 * 1024)))), # 4
        'wc35': tf.Variable(tf.random_normal([1, 1, 1024, 256], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 4
        'wc36': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))), # 4
        'wc37': tf.Variable(tf.random_normal([1, 1, 256, 1024], stddev=np.sqrt(2. / (1 * 1 * 1024)))), # 4
        'wc38': tf.Variable(tf.random_normal([1, 1, 1024, 256], stddev=np.sqrt(2. / (1 * 1 * 256)))), # 4
        'wc39': tf.Variable(tf.random_normal([3, 3, 256, 256], stddev=np.sqrt(2. / (3 * 3 * 256)))), # 4
        'wc40': tf.Variable(tf.random_normal([1, 1, 256, 1024], stddev=np.sqrt(2. / (1 * 1 * 1024)))), # 4

        'wc23a': tf.Variable(tf.random_normal([1, 1, 512, 1024], stddev=np.sqrt(2. / (1 * 1 * 1024)))), # 4a

        'wc41': tf.Variable(tf.random_normal([1, 1, 1024, 512], stddev=np.sqrt(2. / (1 * 1 * 512)))), # 5
        'wc42': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2. / (3 * 3 * 512)))), # 5
        'wc43': tf.Variable(tf.random_normal([1, 1, 512, 2048], stddev=np.sqrt(2. / (1 * 1 * 2048)))), # 5
        'wc44': tf.Variable(tf.random_normal([1, 1, 2048, 512], stddev=np.sqrt(2. / (1 * 1 * 512)))), # 5
        'wc45': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2. / (3 * 3 * 512)))), # 5
        'wc46': tf.Variable(tf.random_normal([1, 1, 512, 2048], stddev=np.sqrt(2. / (1 * 1 * 2048)))), # 5
        'wc47': tf.Variable(tf.random_normal([1, 1, 2048, 512], stddev=np.sqrt(2. / (1 * 1 * 512)))), # 5
        'wc48': tf.Variable(tf.random_normal([3, 3, 512, 512], stddev=np.sqrt(2. / (3 * 3 * 512)))), # 5
        'wc49': tf.Variable(tf.random_normal([1, 1, 512, 2048], stddev=np.sqrt(2. / (1 * 1 * 2048)))), # 5

        'wc41a': tf.Variable(tf.random_normal([1, 1, 1024, 2048], stddev=np.sqrt(2. / (1 * 1 * 2048)))), # 5a

        'wo': tf.Variable(tf.random_normal([7*7*2048, 100], stddev=np.sqrt(2./(7*7*512)))), # out
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

    #  side convolution
    res2_branch = tf.nn.conv2d(pool1, weights['wc2a'], strides=[1, 1, 1, 1], padding='SAME')
    res2_branch = batch_norm_layer(conv2a, train_phase, 'bn2a')
    res2_branch = tf.nn.relu(conv2a)

    # output: 56x56       9 convolutions [1, 1, 64], [3, 3, 64], [1, 1, 256]
    conv2 = tf.nn.conv2d(pool1, weights['wc2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2 = batch_norm_layer(conv2, train_phase, 'bn2')
    conv2 = tf.nn.relu(conv2)

    conv3 = tf.nn.conv2d(conv2, weights['wc3'], strides=[1, 1, 1, 1], padding='SAME')
    conv3 = batch_norm_layer(conv3, train_phase, 'bn3')
    conv3 = tf.nn.relu(conv3)

    conv4 = tf.nn.conv2d(conv3, weights['wc4'], strides=[1, 1, 1, 1], padding='SAME')
    conv4 = batch_norm_layer(conv4, train_phase, 'bn4')
    conv4 = tf.nn.relu(conv4)

    # first residual add
    res2a = res2_branch + conv4
    res2a = tf.nn.relu(res2a)

    conv5 = tf.nn.conv2d(res2a, weights['wc5'], strides=[1, 1, 1, 1], padding='SAME')
    conv5 = batch_norm_layer(conv5, train_phase, 'bn5')
    conv5 = tf.nn.relu(conv5)

    conv6 = tf.nn.conv2d(conv5, weights['wc6'], strides=[1, 1, 1, 1], padding='SAME')
    conv6 = batch_norm_layer(conv6, train_phase, 'bn6')
    conv6 = tf.nn.relu(conv6)

    conv7 = tf.nn.conv2d(conv6, weights['wc7'], strides=[1, 1, 1, 1], padding='SAME')
    conv7 = batch_norm_layer(conv7, train_phase, 'bn7')
    conv7 = tf.nn.relu(conv7)

    # second residual add

    res2b = res2a + conv7
    res2b = tf.nn.relu(res2b)

    conv8 = tf.nn.conv2d(res2b, weights['wc8'], strides=[1, 1, 1, 1], padding='SAME')
    conv8 = batch_norm_layer(conv8, train_phase, 'bn8')
    conv8 = tf.nn.relu(conv8)

    conv9 = tf.nn.conv2d(conv8, weights['wc9'], strides=[1, 1, 1, 1], padding='SAME')
    conv9 = batch_norm_layer(conv9, train_phase, 'bn9')
    conv9 = tf.nn.relu(conv9)

    conv10 = tf.nn.conv2d(conv9, weights['wc10'], strides=[1, 1, 1, 1], padding='SAME')
    conv10 = batch_norm_layer(conv10, train_phase, 'bn10')
    conv10 = tf.nn.relu(conv10)

    # third residual add

    res2c = res2b + conv10
    res2c = tf.nn.relu(res2c)

    # third residual branch

    res3_branch = tf.nn.conv2d(res2c, weights['wc11a'], strides=[1, 2, 2, 1], padding='SAME')
    res3_branch = batch_norm_layer(res3_branch, train_phase, 'bn11a')
    res3_branch = tf.nn.relu(res3_branch)


    # output: 28x28       3x4 convs : [1, 1, 128], [3, 3, 128], [1, 1, 512] x 4, stride on first

    conv11 = tf.nn.conv2d(res2c, weights['wc11'], strides=[1, 2, 2, 1], padding='SAME')
    conv11 = batch_norm_layer(conv11, train_phase, 'bn11')
    conv11 = tf.nn.relu(conv11)

    conv12 = tf.nn.conv2d(conv11, weights['wc12'], strides=[1, 1, 1, 1], padding='SAME')
    conv12 = batch_norm_layer(conv12, train_phase, 'bn12')
    conv12 = tf.nn.relu(conv12)

    conv13 = tf.nn.conv2d(conv12, weights['wc13'], strides=[1, 1, 1, 1], padding='SAME')
    conv13 = batch_norm_layer(conv13, train_phase, 'bn13')
    conv13 = tf.nn.relu(conv13)

    # fourth residual add

    res3a = res3_branch + conv13
    res3a = tf.nn.relu(res3a)

    conv14 = tf.nn.conv2d(res3a, weights['wc14'], strides=[1, 1, 1, 1], padding='SAME')
    conv14 = batch_norm_layer(conv14, train_phase, 'bn14')
    conv14 = tf.nn.relu(conv14)

    conv15 = tf.nn.conv2d(conv14, weights['wc15'], strides=[1, 1, 1, 1], padding='SAME')
    conv15 = batch_norm_layer(conv15, train_phase, 'bn15')
    conv15 = tf.nn.relu(conv15)

    conv16 = tf.nn.conv2d(conv15, weights['wc16'], strides=[1, 1, 1, 1], padding='SAME')
    conv16 = batch_norm_layer(conv16, train_phase, 'bn16')
    conv16 = tf.nn.relu(conv16)

    # fifth residual add

    res3b = res3a + conv16
    res3b = tf.nn.relu(res3b)

    conv17 = tf.nn.conv2d(res3b, weights['wc17'], strides=[1, 1, 1, 1], padding='SAME')
    conv17 = batch_norm_layer(conv17, train_phase, 'bn17')
    conv17 = tf.nn.relu(conv17)

    conv18 = tf.nn.conv2d(conv17, weights['wc18'], strides=[1, 1, 1, 1], padding='SAME')
    conv18 = batch_norm_layer(conv18, train_phase, 'bn18')
    conv18 = tf.nn.relu(conv18)

    conv19 = tf.nn.conv2d(conv18, weights['wc19'], strides=[1, 1, 1, 1], padding='SAME')
    conv19 = batch_norm_layer(conv19, train_phase, 'bn19')
    conv19 = tf.nn.relu(conv19)

    # sixth residual add

    res3c = res3b + conv19
    res3c = tf.nn.relu(res3c)

    conv20 = tf.nn.conv2d(res3c, weights['wc20'], strides=[1, 1, 1, 1], padding='SAME')
    conv20 = batch_norm_layer(conv20, train_phase, 'bn20')
    conv20 = tf.nn.relu(conv20)

    conv21 = tf.nn.conv2d(conv20, weights['wc21'], strides=[1, 1, 1, 1], padding='SAME')
    conv21 = batch_norm_layer(conv21, train_phase, 'bn21')
    conv21 = tf.nn.relu(conv21)

    conv22 = tf.nn.conv2d(conv21, weights['wc22'], strides=[1, 1, 1, 1], padding='SAME')
    conv22 = batch_norm_layer(conv22, train_phase, 'bn22')
    conv22 = tf.nn.relu(conv22)

    # seventh residual add
    res3d = res3c + conv22
    res3d = tf.nn.relu(res3d)

    # build branch for fourth convolutional block

    res4_branch = tf.nn.conv2d(res3d, weights['wc23a'], strides=[1, 2, 2, 1], padding='SAME')
    res4_branch = batch_norm_layer(res4_branch, train_phase, 'bn23a')
    res4_branch = tf.nn.relu(res4_branch)

    
    # output: 14x14       3x6 convs: [1, 1, 256], [3, 3, 256], [1, 1, 1024] x 6, stride on first

    conv23 = tf.nn.conv2d(res3d, weights['wc23'], strides=[1, 2, 2, 1], padding='SAME')
    conv23 = batch_norm_layer(conv23, train_phase, 'bn23')
    conv23 = tf.nn.relu(conv23)

    conv24 = tf.nn.conv2d(conv23, weights['wc24'], strides=[1, 1, 1, 1], padding='SAME')
    conv24 = batch_norm_layer(conv24, train_phase, 'bn24')
    conv24 = tf.nn.relu(conv24)

    conv25 = tf.nn.conv2d(conv24, weights['wc25'], strides=[1, 1, 1, 1], padding='SAME')
    conv25 = batch_norm_layer(conv25, train_phase, 'bn25')
    conv25 = tf.nn.relu(conv25)

    # eigth residual add
    res4a = res4_branch + conv25
    res4a = tf.nn.relu(res4a)

    conv26 = tf.nn.conv2d(res4a, weights['wc26'], strides=[1, 1, 1, 1], padding='SAME')
    conv26 = batch_norm_layer(conv26, train_phase, 'bn26')
    conv26 = tf.nn.relu(conv26)

    conv27 = tf.nn.conv2d(conv26, weights['wc27'], strides=[1, 1, 1, 1], padding='SAME')
    conv27 = batch_norm_layer(conv27, train_phase, 'bn27')
    conv27 = tf.nn.relu(conv27)

    conv28 = tf.nn.conv2d(conv27, weights['wc28'], strides=[1, 1, 1, 1], padding='SAME')
    conv28 = batch_norm_layer(conv28, train_phase, 'bn28')
    conv28 = tf.nn.relu(conv28)

    # ninth residual add

    res4b = res4a + conv28
    res4b = tf.nn.relu(res4b)

    conv29 = tf.nn.conv2d(res4b, weights['wc29'], strides=[1, 1, 1, 1], padding='SAME')
    conv29 = batch_norm_layer(conv29, train_phase, 'bn29')
    conv29 = tf.nn.relu(conv29)

    conv30 = tf.nn.conv2d(conv29, weights['wc30'], strides=[1, 1, 1, 1], padding='SAME')
    conv30 = batch_norm_layer(conv30, train_phase, 'bn30')
    conv30 = tf.nn.relu(conv30)

    conv31 = tf.nn.conv2d(conv30, weights['wc31'], strides=[1, 1, 1, 1], padding='SAME')
    conv31 = batch_norm_layer(conv31, train_phase, 'bn31')
    conv31 = tf.nn.relu(conv31)

    # tenth residual add

    res4c = res4b + conv31
    res4c = tf.nn.relu(res4c)

    conv32 = tf.nn.conv2d(res4c, weights['wc32'], strides=[1, 1, 1, 1], padding='SAME')
    conv32 = batch_norm_layer(conv32, train_phase, 'bn32')
    conv32 = tf.nn.relu(conv32)

    conv33 = tf.nn.conv2d(conv32, weights['wc33'], strides=[1, 1, 1, 1], padding='SAME')
    conv33 = batch_norm_layer(conv33, train_phase, 'bn33')
    conv33 = tf.nn.relu(conv33)

    conv34 = tf.nn.conv2d(conv33, weights['wc34'], strides=[1, 1, 1, 1], padding='SAME')
    conv34 = batch_norm_layer(conv34, train_phase, 'bn34')
    conv34 = tf.nn.relu(conv34)

    # eleventh residual add

    res4d = res4c + conv34
    res4d = tf.nn.relu(res4d)

    conv35 = tf.nn.conv2d(res4d, weights['wc35'], strides=[1, 1, 1, 1], padding='SAME')
    conv35 = batch_norm_layer(conv35, train_phase, 'bn35')
    conv35 = tf.nn.relu(conv35)

    conv36 = tf.nn.conv2d(conv35, weights['wc36'], strides=[1, 1, 1, 1], padding='SAME')
    conv36 = batch_norm_layer(conv36, train_phase, 'bn36')
    conv36 = tf.nn.relu(conv36)

    conv37 = tf.nn.conv2d(conv36, weights['wc37'], strides=[1, 1, 1, 1], padding='SAME')
    conv37 = batch_norm_layer(conv37, train_phase, 'bn37')
    conv37 = tf.nn.relu(conv37)

    # twelth residual add

    res4e = res4d + conv37
    res4e = tf.nn.relu(res4e)

    conv38 = tf.nn.conv2d(res4e, weights['wc38'], strides=[1, 1, 1, 1], padding='SAME')
    conv38 = batch_norm_layer(conv38, train_phase, 'bn38')
    conv38 = tf.nn.relu(conv38)

    conv39 = tf.nn.conv2d(conv38, weights['wc39'], strides=[1, 1, 1, 1], padding='SAME')
    conv39 = batch_norm_layer(conv39, train_phase, 'bn39')
    conv39 = tf.nn.relu(conv39)

    conv40 = tf.nn.conv2d(conv39, weights['wc40'], strides=[1, 1, 1, 1], padding='SAME')
    conv40 = batch_norm_layer(conv40, train_phase, 'bn40')
    conv40 = tf.nn.relu(conv40)

    # thirteenth residual add

    res4f = res4e + conv40
    res4f = tf.nn.relu(res4f)

    # build branch for fifth convolutional block

    res5_branch = tf.nn.conv2d(res4f, weights['wc41a'], strides=[1, 2, 2, 1], padding='SAME')
    res5_branch = batch_norm_layer(res5_branch, train_phase, 'bn41a')
    res5_branch = tf.nn.relu(res5_branch)


    # output: 7x7         3x3 convs: [1, 1, 512], [3, 3, 512], [1, 1, 2048], /2 on first

    conv41 = tf.nn.conv2d(res4f, weights['wc41'], strides=[1, 2, 2, 1], padding='SAME')
    conv41 = batch_norm_layer(conv41, train_phase, 'bn41')
    conv41 = tf.nn.relu(conv41)

    conv42 = tf.nn.conv2d(conv41, weights['wc42'], strides=[1, 1, 1, 1], padding='SAME')
    conv42 = batch_norm_layer(conv42, train_phase, 'bn42')
    conv42 = tf.nn.relu(conv42)

    conv43 = tf.nn.conv2d(conv42, weights['wc43'], strides=[1, 1, 1, 1], padding='SAME')
    conv43 = batch_norm_layer(conv43, train_phase, 'bn43')
    conv43 = tf.nn.relu(conv43)

    # fourteenth residual add

    res5a = res5_branch + conv43
    res5a = tf.nn.relu(res5a)

    conv44 = tf.nn.conv2d(res5a, weights['wc44'], strides=[1, 1, 1, 1], padding='SAME')
    conv44 = batch_norm_layer(conv44, train_phase, 'bn44')
    conv44 = tf.nn.relu(conv44)

    conv45 = tf.nn.conv2d(conv44, weights['wc45'], strides=[1, 1, 1, 1], padding='SAME')
    conv45 = batch_norm_layer(conv45, train_phase, 'bn45')
    conv45 = tf.nn.relu(conv45)

    conv46 = tf.nn.conv2d(conv45, weights['wc46'], strides=[1, 1, 1, 1], padding='SAME')
    conv46 = batch_norm_layer(conv46, train_phase, 'bn46')
    conv46 = tf.nn.relu(conv46)

    # fifteenth residual add

    res5b = res5a + conv46
    res5b = tf.nn.relu(res5b)

    conv47 = tf.nn.conv2d(res5b, weights['wc47'], strides=[1, 1, 1, 1], padding='SAME')
    conv47 = batch_norm_layer(conv47, train_phase, 'bn47')
    conv47 = tf.nn.relu(conv47)

    conv48 = tf.nn.conv2d(conv47, weights['wc48'], strides=[1, 1, 1, 1], padding='SAME')
    conv48 = batch_norm_layer(conv48, train_phase, 'bn48')
    conv48 = tf.nn.relu(conv48)

    conv49 = tf.nn.conv2d(conv48, weights['wc49'], strides=[1, 1, 1, 1], padding='SAME')
    conv49 = batch_norm_layer(conv49, train_phase, 'bn49')
    conv49 = tf.nn.relu(conv49)

    # sixteenth residual add

    res5c = res5b + conv49
    res5c = tf.nn.relu(res5c)

    # output: 7x7         1 avg pooling
    pool2 = tf.nn.avg_pool(res5c, ksize=[1, 3, 3, 1], strides=[1,1,1,1], padding='SAME')

    # output: 100         1 fully-connected layer output 100 (7*7*2048, 100)
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
logits = resnet_50(x, train_phase)

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
    
    step = 0

    while step < training_iters:
        # Load a batch of training data
        images_batch, labels_batch = loader_train.next_batch(batch_size)
        
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
    outpt = open('allenlee.resnet50.pred.txt', 'w')
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
        # print (test_img_lab)
        outpt.write(test_img_lab + "\n")
    outpt.close()



