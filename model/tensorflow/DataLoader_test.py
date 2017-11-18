import os
import numpy as np
import scipy.misc
import h5py
np.random.seed(123)


# Loading data from disk
class DataLoaderDisk_t(object):
    def __init__(self, **kwargs):

        self.load_size = int(kwargs['load_size'])
        self.fine_size = int(kwargs['fine_size'])
        self.data_mean = np.array(kwargs['data_mean'])
        self.randomize = kwargs['randomize']
        self.data_root = os.path.join(kwargs['data_root'])

        # read data info from lists
        self.list_im = []

        with open(kwargs['data_list'], 'r') as f:
            for line in f:
                path =line.rstrip()
                self.list_im.append(os.path.join(self.data_root, path))

        self.list_im = np.array(self.list_im, np.object)
        self.num = self.list_im.shape[0]
        print('# Images found:', self.num)


        self._idx = 0
        
    def next_batch(self, batch_size):
        images_batch = np.zeros((batch_size, self.fine_size, self.fine_size, 3)) 
        for i in range(batch_size):
            image = scipy.misc.imread(self.list_im[self._idx])
            image = scipy.misc.imresize(image, (self.load_size, self.load_size))
            image = image.astype(np.float32)/255.
            image = image - self.data_mean
            if self.randomize:
                flip = np.random.random_integers(0, 1)
                if flip>0:
                    image = image[:,::-1,:]
                offset_h = np.random.random_integers(0, self.load_size-self.fine_size)
                offset_w = np.random.random_integers(0, self.load_size-self.fine_size)
            else:
                offset_h = (self.load_size-self.fine_size)//2
                offset_w = (self.load_size-self.fine_size)//2

            images_batch[i, ...] =  image[offset_h:offset_h+self.fine_size, offset_w:offset_w+self.fine_size, :]
            
            self._idx += 1
            if self._idx == self.num:
                self._idx = 0
        
        return images_batch
    
    def size(self):
        return self.num

    def reset(self):
        self._idx = 0
